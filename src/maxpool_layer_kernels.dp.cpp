//#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <dpct/blas_utils.hpp>

#include "maxpool_layer.h"
#include "darknet_cuda.h"

void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output, int *indexes,
                                  const sycl::nd_item<3> &item_ct1)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *delta, float *prev_delta, int *indexes,
                                   const sycl::nd_item<3> &item_ct1)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;
    int area = (size-1)/stride;

    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if(id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    float d = 0;
    int l, m;
    for(l = -area; l < area+1; ++l){
        for(m = -area; m < area+1; ++m){
            int out_w = (j-w_offset)/stride + m;
            int out_h = (i-h_offset)/stride + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}

extern "C" void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    int h = layer.out_h;
    int w = layer.out_w;
    int c = layer.c;

    size_t n = h*w*c*layer.batch;

    /*
    DPCT1049:54: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            forward_maxpool_layer_kernel(n, layer.h, layer.w, layer.c,
                                         layer.stride, layer.size, layer.pad,
                                         net.input_gpu, layer.output_gpu,
                                         layer.indexes_gpu, item_ct1);
        });
    /*
    DPCT1010:121: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
    size_t n = layer.h*layer.w*layer.c*layer.batch;

    /*
    DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            backward_maxpool_layer_kernel(n, layer.h, layer.w, layer.c,
                                          layer.stride, layer.size, layer.pad,
                                          layer.delta_gpu, net.delta_gpu,
                                          layer.indexes_gpu, item_ct1);
        });
    /*
    DPCT1010:122: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

