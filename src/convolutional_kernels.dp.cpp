#include <dpct/dnnl_utils.hpp>
#include <cmath>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <dpct/blas_utils.hpp>

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "darknet_cuda.h"
}

void binarize_kernel(float *x, int n, float *binary,
                     const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= n) return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            binarize_kernel(x, n, binary, item_ct1);
        });
    /*
    DPCT1010:63: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void binarize_input_kernel(float *input, int n, int size, float *binary,
                           const sycl::nd_item<3> &item_ct1)
{
    int s = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += sycl::fabs(input[i * size + s]);
    }
    mean = mean / n;
    for(i = 0; i < n; ++i){
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            binarize_input_kernel(input, n, size, binary, item_ct1);
        });
    /*
    DPCT1010:64: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}


void binarize_weights_kernel(float *weights, int n, int size, float *binary,
                             const sycl::nd_item<3> &item_ct1)
{
    int f = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (f >= n) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < size; ++i){
        mean += sycl::fabs(weights[f * size + i]);
    }
    mean = mean / size;
    for(i = 0; i < size; ++i){
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        //binary[f*size + i] = weights[f*size + i];
    }
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            binarize_weights_kernel(weights, n, size, binary, item_ct1);
        });
    /*
    DPCT1010:65: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.binary){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if(l.xnor){
        binarize_weights_gpu(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }

#ifdef CUDNN
    float one = 1;
    cudnn_handle().async_convolution_forward(
        *((dpct::dnnl::convolution_desc*)l.convDesc), 
        *((dnnl::algorithm)l.fw_algo), one, 
        *((dpct::dnnl::memory_desc_ext*)l.srcTensorDesc), net.input_gpu,
        *((dpct::dnnl::memory_desc_ext*)*((dpct::dnnl::memory_desc_ext*)(l.weightDesc))), l.weights_gpu, one, 
        *((dpct::dnnl::memory_desc_ext*)l.dstTensorDesc), l.output_gpu);

#else
    int i, j;
    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.weights_gpu + j*l.nweights/l.groups;
            float *b = net.workspace;
            float *c = l.output_gpu + (i*l.groups + j)*n*m;
            float *im = net.input_gpu + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1){
                b = im;
            } else {
                im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm_gpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }
#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
    //if(l.dot > 0) dot_error_gpu(l);
    if(l.binary || l.xnor) swap_binary(&l);
}

void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta,
                   const sycl::nd_item<3> &item_ct1)
{
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

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    /*
    DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            smooth_kernel(l.output_gpu, n, l.w, l.h, l.c, size, rate,
                          l.delta_gpu, item_ct1);
        });
    /*
    DPCT1010:66: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
    if(l.smooth){
        smooth_layer(l, 5, l.smooth);
    }
    //constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;

    if(l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
    float one = 1;
    cudnn_handle().async_convolution_backward_weight(
        *((dpct::dnnl::convolution_desc*)l.convDesc), 
        *((dnnl::algorithm)*((dnnl::algorithm*)(l.bf_algo))), one, 
        *((dpct::dnnl::memory_desc_ext*)l.srcTensorDesc), net.input_gpu,
        *((dpct::dnnl::memory_desc_ext*)l.ddstTensorDesc), l.delta_gpu, one, 
        *((dpct::dnnl::memory_desc_ext*)l.dweightDesc), l.weight_updates_gpu);

    if(net.delta_gpu){
        if(l.binary || l.xnor) swap_binary(&l);
        cudnn_handle().async_convolution_backward_data(
            *((dpct::dnnl::convolution_desc*)(l.convDesc)), 
            *((dnnl::algorithm*)(l.bd_algo)), one, 
            *((dpct::dnnl::memory_desc_ext*)(l.weightDesc)), 
            l.weights_gpu,
            *((dpct::dnnl::memory_desc_ext*)(l.ddstTensorDesc)), l.delta_gpu, one, 
            *((dpct::dnnl::memory_desc_ext*)(l.dsrcTensorDesc)), net.delta_gpu);
        if(l.binary || l.xnor) swap_binary(&l);
        if(l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
    }

#else
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    int i, j;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            float *a = l.delta_gpu + (i*l.groups + j)*m*k;
            float *b = net.workspace;
            float *c = l.weight_updates_gpu + j*l.nweights/l.groups;

            float *im  = net.input_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;
            float *imd = net.delta_gpu+(i*l.groups + j)*l.c/l.groups*l.h*l.w;

            im2col_gpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            gemm_gpu(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta_gpu) {
                if (l.binary || l.xnor) swap_binary(&l);
                a = l.weights_gpu + j*l.nweights/l.groups;
                b = l.delta_gpu + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm_gpu(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_gpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
                if(l.binary || l.xnor) {
                    swap_binary(&l);
                }
            }
            if(l.xnor) gradient_array_gpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, net.delta_gpu + i*l.c*l.h*l.w);
        }
    }
#endif
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
    if(l.clip){
        constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}


