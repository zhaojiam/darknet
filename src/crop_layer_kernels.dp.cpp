#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <dpct/blas_utils.hpp>

extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "darknet_cuda.h"
#include "image.h"
}

float get_pixel_kernel(float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

sycl::float3 rgb_to_hsv_kernel(sycl::float3 rgb)
{
    float r = rgb.x();
    float g = rgb.y();
    float b = rgb.z();

    float h, s, v;
    float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max - min;
    v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return sycl::float3(h, s, v);
}

sycl::float3 hsv_to_rgb_kernel(sycl::float3 hsv)
{
    float h = hsv.x();
    float s = hsv.y();
    float v = hsv.z();

    float r, g, b;
    float f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int)sycl::floor(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return sycl::float3(r, g, b);
}

float bilinear_interpolate_kernel(float *image, int w, int h, float x, float y, int c)
{
    int ix = (int)sycl::floor(x);
    int iy = (int)sycl::floor(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

void levels_image_kernel(float *image, float *rand, int batch, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift,
                         const sycl::nd_item<3> &item_ct1)
{
    int size = batch * w * h;
    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    float rshift = rand[0];
    float gshift = rand[1];
    float bshift = rand[2];
    float r0 = rand[8*id + 0];
    float r1 = rand[8*id + 1];
    float r2 = rand[8*id + 2];
    float r3 = rand[8*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5f) ? 1.f/saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5f) ? 1.f/exposure : exposure;

    size_t offset = id * h * w * 3;
    image += offset;
    float r = image[x + w*(y + h*0)];
    float g = image[x + w*(y + h*1)];
    float b = image[x + w*(y + h*2)];
    sycl::float3 rgb = sycl::float3(r, g, b);
    if(train){
        sycl::float3 hsv = rgb_to_hsv_kernel(rgb);
        hsv.y() *= saturation;
        hsv.z() *= exposure;
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    image[x + w * (y + h * 0)] =
        rgb.x() * scale + translate + (rshift - .5f) * shift;
    image[x + w * (y + h * 1)] =
        rgb.y() * scale + translate + (gshift - .5f) * shift;
    image[x + w * (y + h * 2)] =
        rgb.z() * scale + translate + (bshift - .5f) * shift;
}

void forward_crop_layer_kernel(float *input, float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle, float *output,
                               const sycl::nd_item<3> &item_ct1)
{
    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if(id >= size) return;

    float cx = w/2.f;
    float cy = h/2.f;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    float r4 = rand[8*b + 4];
    float r5 = rand[8*b + 5];
    float r6 = rand[8*b + 6];
    float r7 = rand[8*b + 7];

    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5f));
    angle = 2*angle*r7 - angle;
    if(!train){
        dw = (w - crop_width)/2.f;
        dh = (h - crop_height)/2.f;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    float x = (flip) ? w - dw - j - 1 : j + dw;    
    float y = i + dh;

    float rx = sycl::cos(angle) * (x - cx) - sycl::sin(angle) * (y - cy) + cx;
    float ry = sycl::sin(angle) * (x - cx) + sycl::cos(angle) * (y - cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}

extern "C" void forward_crop_layer_gpu(crop_layer layer, network net)
{
    cuda_random(layer.rand_gpu, layer.batch*8);

    float radians = layer.angle*3.14159265f/180.f;

    float scale = 2;
    float translate = -1;
    if(layer.noadjust){
        scale = 1;
        translate = 0;
    }

    int size = layer.batch * layer.w * layer.h;

    /*
    DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            levels_image_kernel(net.input_gpu, layer.rand_gpu, layer.batch,
                                layer.w, layer.h, net.train, layer.saturation,
                                layer.exposure, translate, scale, layer.shift,
                                item_ct1);
        });
    /*
    DPCT1010:78: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);

    size = layer.batch*layer.c*layer.out_w*layer.out_h;

    /*
    DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            forward_crop_layer_kernel(net.input_gpu, layer.rand_gpu, size,
                                      layer.c, layer.h, layer.w, layer.out_h,
                                      layer.out_w, net.train, layer.flip,
                                      radians, layer.output_gpu, item_ct1);
        });
    /*
    DPCT1010:79: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);

/*
       cuda_pull_array(layer.output_gpu, layer.output, size);
       image im = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 0*(size/layer.batch));
       image im2 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 1*(size/layer.batch));
       image im3 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 2*(size/layer.batch));

       translate_image(im, -translate);
       scale_image(im, 1/scale);
       translate_image(im2, -translate);
       scale_image(im2, 1/scale);
       translate_image(im3, -translate);
       scale_image(im3, 1/scale);
       
       show_image(im, "cropped");
       show_image(im2, "cropped2");
       show_image(im3, "cropped3");
       cvWaitKey(0);
       */
}

