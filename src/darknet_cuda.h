#ifndef CUDA_H
#define CUDA_H

#ifdef __cpluscplus
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#endif

#include "darknet.h"

#if defined(GPU) && defined(__cplusplus)
void check_error(dpct::err0 status);
dpct::blas::descriptor_ptr blas_handle();
sycl::range<3> cuda_gridsize(size_t n);
#endif

#ifdef GPU

int *cuda_make_int_array(int *x, size_t n);
void cuda_random(float *x_gpu, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);


#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
