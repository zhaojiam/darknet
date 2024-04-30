#ifndef CUDA_H
#define CUDA_H


#include "darknet.h"

#if defined(GPU) && defined(__cplusplus)
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

extern "C" {
// SYCL: This function is only invoked by C++ files when MACRO GPU is set.
// SYCL: But when C++ file declared these function using a extern "C". Bad practice :(
void check_error(dpct::err0 status); 
dpct::blas::descriptor_ptr blas_handle();
sycl::range<3> cuda_gridsize(size_t n);
void cuda_random(float *x_gpu, size_t n);

#endif

#ifdef GPU
// SYCL: This function is invoked by both C and C++ files when Macro GPU is set.
int *cuda_make_int_array(int *x, size_t n);
#endif


#if defined(GPU) && defined(__cplusplus)
} // end of extern C
#endif


#if defined(GPU) && defined(CUDNN) && defined(__cplusplus)

#include <dpct/dnnl_utils.hpp>
dpct::dnnl::engine_ext cudnn_handle();

#endif

#endif