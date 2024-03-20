#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <dpct/blas_utils.hpp>
#include <assert.h>

extern "C" {
#include "blas.h"
#include "darknet_cuda.h"
#include "utils.h"
}

void scale_bias_kernel(float *output, float *biases, int n, int size,
                       const sycl::nd_item<3> &item_ct1)
{
    int offset = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);
    int filter = item_ct1.get_group(1);
    int batch = item_ct1.get_group(0);

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    sycl::range<3> dimGrid(batch, n, (size - 1) / BLOCK + 1);
    sycl::range<3> dimBlock(1, 1, BLOCK);

    /*
    DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
        [=](sycl::nd_item<3> item_ct1) {
            scale_bias_kernel(output, biases, n, size, item_ct1);
        });
    /*
    DPCT1010:88: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates,
                           const sycl::nd_item<3> &item_ct1, float *part)
{

    int i,b;
    int filter = item_ct1.get_group(2);
    int p = item_ct1.get_local_id(2);
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    /*
    DPCT1065:89: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    /*
    DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:138: 'BLOCK' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> part_acc_ct1(
            sycl::range<1>(512 /*BLOCK*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, n) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                backward_scale_kernel(
                    x_norm, delta, batch, n, size, scale_updates, item_ct1,
                    part_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:90: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void add_bias_kernel(float *output, float *biases, int batch, int n, int size,
                     const sycl::nd_item<3> &item_ct1)
{
    int index = (item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                    item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    /*
    DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(num) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            add_bias_kernel(output, biases, batch, n, size, item_ct1);
        });
    /*
    DPCT1010:91: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n,
                               const sycl::nd_item<3> &item_ct1)
{
    int index = (item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                    item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}

void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size,
                          const sycl::nd_item<3> &item_ct1, float *part)
{

    int i,b;
    int filter = item_ct1.get_group(2);
    int p = item_ct1.get_local_id(2);
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    /*
    DPCT1065:92: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    if(size == 1){
        /*
        DPCT1049:21: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().parallel_for(
            sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                backward_bias_conn_kernel(bias_updates, delta, batch, n,
                                          item_ct1);
            });
    }else{
        /*
        DPCT1049:22: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:139: 'BLOCK' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<float, 1> part_acc_ct1(
                sycl::range<1>(512 /*BLOCK*/), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, n) *
                                      sycl::range<3>(1, 1, BLOCK),
                                  sycl::range<3>(1, 1, BLOCK)),
                [=](sycl::nd_item<3> item_ct1) {
                    backward_bias_kernel(
                        bias_updates, delta, batch, n, size, item_ct1,
                        part_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    }
    /*
    DPCT1010:93: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

/*
__global__ void dot_kernel(float *output, float scale, int batch, int n, int size, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int f1 = index / n;
    int f2 = index % n;
    if (f2 <= f1) return;
    
    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    int b, i;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            sum += output[i1] * output[i2];
            norm1 += output[i1] * output[i1];
            norm2 += output[i2] * output[i2];
        }
    }
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);
    float norm = norm1 * norm2;
    sum = sum / norm;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            delta[i1] += - scale * sum * output[i2] / norm;
            delta[i2] += - scale * sum * output[i1] / norm;
        }
    }
}

void dot_error_gpu(layer l)
{
    dot_kernel<<<cuda_gridsize(l.n*l.n), BLOCK>>>(l.output_gpu, l.dot, l.batch, l.n, l.out_w * l.out_h, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}
*/


void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t,
                 const sycl::nd_item<3> &item_ct1)
{
    int index = (item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                    item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= N) return;

    float mhat = m[index] / (1.f - dpct::pow(B1, t));
    float vhat = v[index] / (1.f - dpct::pow(B2, t));

    x[index] = x[index] + rate * mhat / (sycl::sqrt(vhat) + eps);
}

extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    /*
    DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            adam_kernel(n, x, m, v, B1, B2, rate, eps, t, item_ct1);
        });
    /*
    DPCT1010:94: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, (1-B1), d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
    fill_gpu(n, 0, d, 1);
}

void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial,
                      const sycl::nd_item<3> &item_ct1)
{
    int index = (item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                    item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= N) return;
    int f = (index/spatial)%filters;

    x[index] = (x[index] - mean[f]) / (sycl::sqrt(variance[f] + .00001f));
}

void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta,
                            const sycl::nd_item<3> &item_ct1)
{
    int index = (item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                    item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= N) return;
    int f = (index/spatial)%filters;

    delta[index] =
        delta[index] * 1.f / (sycl::sqrt(variance[f] + .00001f)) +
        variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) +
        mean_delta[f] / (spatial * batch);
}

extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    /*
    DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            normalize_delta_kernel(N, x, mean, variance, mean_delta,
                                   variance_delta, batch, filters, spatial,
                                   delta, item_ct1);
        });
    /*
    DPCT1010:95: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void  variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta,
                            const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *=
        -.5f * dpct::pow(variance[i] + .00001f, (float)(-3.f / 2.f));
}

void accumulate_kernel(float *x, int n, int groups, float *sum,
                       const sycl::nd_item<3> &item_ct1)
{
    int k;
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta,
                            const sycl::nd_item<3> &item_ct1, float *local)
{
    const int threads = BLOCK;

    int id = item_ct1.get_local_id(2);
    local[id] = 0;

    int filter = item_ct1.get_group(2);

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    /*
    DPCT1065:96: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f / sycl::sqrt(variance[filter] + .00001f));
    }
}

void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta,
                                 const sycl::nd_item<3> &item_ct1, float *local)
{
    const int threads = BLOCK;

    int id = item_ct1.get_local_id(2);
    local[id] = 0;

    int filter = item_ct1.get_group(2);

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    /*
    DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *=
            -.5f * dpct::pow(variance[filter] + .00001f, (float)(-3.f / 2.f));
    }
}


void mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta,
                       const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f / sycl::sqrt(variance[i] + .00001f));
}

extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    /*
    DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(filters) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            mean_delta_kernel(delta, variance, batch, filters, spatial,
                              mean_delta, item_ct1);
        });
    /*
    DPCT1010:98: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    /*
    DPCT1049:26: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:140: 'threads' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> local_acc_ct1(
            sycl::range<1>(512 /*threads*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, filters) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                fast_mean_delta_kernel(
                    delta, variance, batch, filters, spatial, mean_delta,
                    item_ct1,
                    local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:99: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    /*
    DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:141: 'threads' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> local_acc_ct1(
            sycl::range<1>(512 /*threads*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, filters) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                fast_variance_delta_kernel(
                    x, delta, mean, variance, batch, filters, spatial,
                    variance_delta, item_ct1,
                    local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:100: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean,
                  const sycl::nd_item<3> &item_ct1)
{
    float scale = 1.f/(batch * spatial);
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance,
                     const sycl::nd_item<3> &item_ct1)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += (x[index] - mean[i]) * (x[index] - mean[i]);
        }
    }
    variance[i] *= scale;
}

void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out,
                  const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY,
                const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) Y[i * INCY] = dpct::pow(X[i * INCX], ALPHA);
}

void const_kernel(int N, float ALPHA, float *X, int INCX,
                  const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) X[i*INCX] = ALPHA;
}

void constrain_kernel(int N, float ALPHA, float *X, int INCX,
                      const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if (i < N) X[i * INCX] = sycl::fmin(ALPHA, sycl::fmax(-ALPHA, X[i * INCX]));
}

void supp_kernel(int N, float ALPHA, float *X, int INCX,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}

void add_kernel(int N, float ALPHA, float *X, int INCX,
                const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) X[i*INCX] += ALPHA;
}

void scal_kernel(int N, float ALPHA, float *X, int INCX,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) X[i*INCX] *= ALPHA;
}

void fill_kernel(int N, float ALPHA, float *X, int INCX,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) X[i*INCX] = ALPHA;
}

void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

void mul_kernel(int N, float *X, int INCX, float *Y, int INCY,
                const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < N) Y[i*INCY] *= X[i*INCX];
}


extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    /*
    DPCT1049:28: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            normalize_kernel(N, x, mean, variance, batch, filters, spatial,
                             item_ct1);
        });
    /*
    DPCT1010:101: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void l2norm_kernel(int N, float *x, float *dx, int batch, int filters, int spatial,
                   const sycl::nd_item<3> &item_ct1)
{
    int index = (item_ct1.get_group(2) +
                 item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                    item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += x[index] * x[index];
    }
    sum = sycl::sqrt(sum);
    if(sum == 0) sum = 1;
    //printf("%f\n", sum);
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (1 - x[index]) / sum;
    }
}

extern "C" void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial)
{
    size_t N = batch*spatial;
    /*
    DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            l2norm_kernel(N, x, dx, batch, filters, spatial, item_ct1);
        });
    /*
    DPCT1010:102: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void  fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean,
                       const sycl::nd_item<3> &item_ct1, float *local)
{
    const int threads = BLOCK;

    int id = item_ct1.get_local_id(2);
    local[id] = 0;

    int filter = item_ct1.get_group(2);

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    /*
    DPCT1065:103: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance,
                           const sycl::nd_item<3> &item_ct1, float *local)
{
    const int threads = BLOCK;

    int id = item_ct1.get_local_id(2);
    local[id] = 0;

    int filter = item_ct1.get_group(2);

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i + id < spatial) ? (x[index] - mean[filter]) *
                                                  (x[index] - mean[filter])
                                            : 0;
        }
    }

    /*
    DPCT1065:104: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    /*
    DPCT1049:30: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:142: 'threads' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> local_acc_ct1(
            sycl::range<1>(512 /*threads*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, filters) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                fast_mean_kernel(
                    x, batch, filters, spatial, mean, item_ct1,
                    local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:105: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    /*
    DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        /*
        DPCT1101:143: 'threads' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<float, 1> local_acc_ct1(
            sycl::range<1>(512 /*threads*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, filters) *
                                  sycl::range<3>(1, 1, BLOCK),
                              sycl::range<3>(1, 1, BLOCK)),
            [=](sycl::nd_item<3> item_ct1) {
                fast_variance_kernel(
                    x, mean, batch, filters, spatial, variance, item_ct1,
                    local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>()
                        .get());
            });
    });
    /*
    DPCT1010:106: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}


extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    /*
    DPCT1049:32: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(filters) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            mean_kernel(x, batch, filters, spatial, mean, item_ct1);
        });
    /*
    DPCT1010:107: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    /*
    DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(filters) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            variance_kernel(x, mean, batch, filters, spatial, variance,
                            item_ct1);
        });
    /*
    DPCT1010:108: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    /*
    DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            pow_kernel(N, ALPHA, X, INCX, Y, INCY, item_ct1);
        });
    /*
    DPCT1010:109: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    /*
    DPCT1049:35: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            axpy_kernel(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY, item_ct1);
        });
    /*
    DPCT1010:110: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    /*
    DPCT1049:36: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            mul_kernel(N, X, INCX, Y, INCY, item_ct1);
        });
    /*
    DPCT1010:111: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    /*
    DPCT1049:37: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            copy_kernel(N, X, OFFX, INCX, Y, OFFY, INCY, item_ct1);
        });
    /*
    DPCT1010:112: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out,
                    const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}

extern "C" void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int size = spatial*batch*layers;
    /*
    DPCT1049:38: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            flatten_kernel(size, x, spatial, layers, batch, forward, out,
                           item_ct1);
        });
    /*
    DPCT1010:113: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int size = w*h*c*batch;
    /*
    DPCT1049:39: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            reorg_kernel(size, x, w, h, c, batch, stride, forward, out,
                         item_ct1);
        });
    /*
    DPCT1010:114: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void mask_kernel(int n,  float *x, float mask_num, float *mask, float val,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n && mask[i] == mask_num) x[i] = val;
}

extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask, float val)
{
    /*
    DPCT1049:40: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            mask_kernel(N, X, mask_num, mask, val, item_ct1);
        });
    /*
    DPCT1010:115: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void scale_mask_kernel(int n,  float *x, float mask_num, float *mask, float scale,
                       const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}

extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
{
    /*
    DPCT1049:41: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_mask_kernel(N, X, mask_num, mask, scale, item_ct1);
        });
    /*
    DPCT1010:116: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
{
    /*
    DPCT1049:42: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            const_kernel(N, ALPHA, X, INCX, item_ct1);
        });
    /*
    DPCT1010:117: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void constrain_gpu(int N, float ALPHA, float * X, int INCX)
{
    /*
    DPCT1049:43: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            constrain_kernel(N, ALPHA, X, INCX, item_ct1);
        });
    /*
    DPCT1010:118: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}


extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
{
    /*
    DPCT1049:44: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            add_kernel(N, ALPHA, X, INCX, item_ct1);
        });
    /*
    DPCT1010:119: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
{
    /*
    DPCT1049:45: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            scal_kernel(N, ALPHA, X, INCX, item_ct1);
        });
    /*
    DPCT1010:120: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void supp_gpu(int N, float ALPHA, float * X, int INCX)
{
    /*
    DPCT1049:46: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            supp_kernel(N, ALPHA, X, INCX, item_ct1);
        });
    /*
    DPCT1010:121: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    /*
    DPCT1049:47: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(N) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            fill_kernel(N, ALPHA, X, INCX, item_ct1);
        });
    /*
    DPCT1010:122: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out,
                     const sycl::nd_item<3> &item_ct1)
{
    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
    //out[out_index] += add[add_index];
}

extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    /*
    DPCT1049:48: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            shortcut_kernel(size, minw, minh, minc, stride, sample, batch, w1,
                            h1, c1, add, w2, h2, c2, s1, s2, out, item_ct1);
        });
    /*
    DPCT1010:123: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error,
                      const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = sycl::fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    /*
    DPCT1049:49: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            smooth_l1_kernel(n, pred, truth, delta, error, item_ct1);
        });
    /*
    DPCT1010:124: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error,
                          const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -sycl::log(p) : 0;
        delta[i] = t-p;
    }
}

extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    /*
    DPCT1049:50: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            softmax_x_ent_kernel(n, pred, truth, delta, error, item_ct1);
        });
    /*
    DPCT1010:125: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void logistic_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error,
                           const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t * sycl::log(p + .0000001) -
                   (1 - t) * sycl::log(1 - p + .0000001);
        delta[i] = t-p;
    }
}

extern "C" void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    /*
    DPCT1049:51: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            logistic_x_ent_kernel(n, pred, truth, delta, error, item_ct1);
        });
    /*
    DPCT1010:126: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void l2_kernel(int n, float *pred, float *truth, float *delta, float *error,
               const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = diff;
    }
}

extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    /*
    DPCT1049:52: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            l2_kernel(n, pred, truth, delta, error, item_ct1);
        });
    /*
    DPCT1010:127: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void l1_kernel(int n, float *pred, float *truth, float *delta, float *error,
               const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = sycl::fabs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}

extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    /*
    DPCT1049:53: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            l1_kernel(n, pred, truth, delta, error, item_ct1);
        });
    /*
    DPCT1010:128: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void wgan_kernel(int n, float *pred, float *truth, float *delta, float *error,
                 const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        error[i] = truth[i] ? -pred[i] : pred[i];
        delta[i] = (truth[i] > 0) ? 1 : -1;
    }
}

extern "C" void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    /*
    DPCT1049:54: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(n) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            wgan_kernel(n, pred, truth, delta, error, item_ct1);
        });
    /*
    DPCT1010:129: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}




void weighted_sum_kernel(int n, float *a, float *b, float *s, float *c,
                         const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

void deinter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT,
                    const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}

extern "C" void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    /*
    DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize((NX + NY) * B) *
                              sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            deinter_kernel(NX, X, NY, Y, B, OUT, item_ct1);
        });
    /*
    DPCT1010:130: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void inter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT,
                  const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}

extern "C" void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    /*
    DPCT1049:56: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize((NX + NY) * B) *
                              sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            inter_kernel(NX, X, NY, Y, B, OUT, item_ct1);
        });
    /*
    DPCT1010:131: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
{
    /*
    DPCT1049:57: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(num) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            weighted_sum_kernel(num, a, b, s, c, item_ct1);
        });
    /*
    DPCT1010:132: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void weighted_delta_kernel(int n, float *a, float *b, float *s, float *da, float *db, float *ds, float *dc,
                           const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
{
    /*
    DPCT1049:58: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(num) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            weighted_delta_kernel(num, a, b, s, da, db, ds, dc, item_ct1);
        });
    /*
    DPCT1010:133: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void mult_add_into_kernel(int n, float *a, float *b, float *c,
                          const sycl::nd_item<3> &item_ct1)
{
    int i = (item_ct1.get_group(2) +
             item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);
    if(i < n){
        c[i] += a[i]*b[i];
    }
}

extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
{
    /*
    DPCT1049:59: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(num) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            mult_add_into_kernel(num, a, b, c, item_ct1);
        });
    /*
    DPCT1010:134: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}


void softmax_device(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = sycl::native::exp(input[i * stride] / temp - largest / temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset,
                         const sycl::nd_item<3> &item_ct1)
{
    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

extern "C" void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier)
{
    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
    /*
       static int *tree_groups_size = 0;
       static int *tree_groups_offset = 0;
       if(!tree_groups_size){
       tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
       tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
       }
     */
    int num = spatial*batch*hier.groups;
    /*
    DPCT1049:60: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(num) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            softmax_tree_kernel(input, spatial, batch, stride, temp, output,
                                hier.groups, tree_groups_size,
                                tree_groups_offset, item_ct1);
        });
    /*
    DPCT1010:135: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
    cuda_free((float *)tree_groups_size);
    cuda_free((float *)tree_groups_offset);
}

void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output,
                    const sycl::nd_item<3> &item_ct1)
{
    int id = (item_ct1.get_group(2) +
              item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                 item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    /*
    DPCT1049:61: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(batch * groups) *
                              sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            softmax_kernel(input, n, batch, batch_offset, groups, group_offset,
                           stride, temp, output, item_ct1);
        });
    /*
    DPCT1010:136: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}


void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out,
                     const sycl::nd_item<3> &item_ct1)
{
    size_t i = (item_ct1.get_group(2) +
                item_ct1.get_group(1) * item_ct1.get_group_range(2)) *
                   item_ct1.get_local_range(2) +
               item_ct1.get_local_id(2);
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    else dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        x + in_index, scale * out[out_index]);
}
extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    /*
    DPCT1049:62: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(cuda_gridsize(size) * sycl::range<3>(1, 1, BLOCK),
                          sycl::range<3>(1, 1, BLOCK)),
        [=](sycl::nd_item<3> item_ct1) {
            upsample_kernel(size, in, w, h, c, batch, stride, forward, scale,
                            out, item_ct1);
        });
    /*
    DPCT1010:137: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}
