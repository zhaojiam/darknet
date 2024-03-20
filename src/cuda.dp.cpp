int gpu_index = 0;

#ifdef GPU

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "darknet_cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void cuda_set_device(int n) try {
    gpu_index = n;
    /*
    DPCT1093:69: The "n" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    dpct::err0 status = DPCT_CHECK_ERROR(dpct::select_device(n));
    check_error(status);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int cuda_get_device() try {
    int n = 0;
    dpct::err0 status =
        DPCT_CHECK_ERROR(n = dpct::dev_mgr::instance().current_device_id());
    check_error(status);
    return n;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void check_error(dpct::err0 status)
{
    //cudaDeviceSynchronize();
    /*
    DPCT1010:74: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 status2 = 0;
    /*
    DPCT1000:71: Error handling if-stmt was detected but could not be rewritten.
    */
    if (status != 0)
    {
        /*
        DPCT1009:75: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced by a placeholder string. You need to
        rewrite this code.
        */
        const char *s = "<Placeholder string>";
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        /*
        DPCT1001:70: The statement could not be removed.
        */
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    /*
    DPCT1000:73: Error handling if-stmt was detected but could not be rewritten.
    */
    if (status2 != 0)
    {
        /*
        DPCT1009:76: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced by a placeholder string. You need to
        rewrite this code.
        */
        const char *s = "<Placeholder string>";
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        /*
        DPCT1001:72: The statement could not be removed.
        */
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

sycl::range<3> cuda_gridsize(size_t n) {
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    sycl::range<3> d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

dpct::blas::descriptor_ptr blas_handle()
{
    static int init[16] = {0};
    static dpct::blas::descriptor_ptr handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        handle[i] = new dpct::blas::descriptor();
        init[i] = 1;
    }
    return handle[i];
}

float *cuda_make_array(float *x, size_t n) try {
    float *x_gpu;
    size_t size = sizeof(float)*n;
    dpct::err0 status = DPCT_CHECK_ERROR(
        x_gpu = (float *)sycl::malloc_device(size, dpct::get_in_order_queue()));
    check_error(status);
    if(x){
        status = DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(x_gpu, x, size).wait());
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void cuda_random(float *x_gpu, size_t n)
{
    static dpct::rng::host_rng_ptr gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        *(&gen[i]) =
            dpct::rng::create_host_rng(dpct::rng::random_engine_type::mcg59);
        gen[i]->set_seed(time(0));
        init[i] = 1;
    }
    gen[i]->generate_uniform(x_gpu, n);
    /*
    DPCT1010:77: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = (float *)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n) try {
    int *x_gpu;
    size_t size = sizeof(int)*n;
    dpct::err0 status = DPCT_CHECK_ERROR(
        x_gpu = (int *)sycl::malloc_device(size, dpct::get_in_order_queue()));
    check_error(status);
    if(x){
        status = DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(x_gpu, x, size).wait());
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void cuda_free(float *x_gpu) try {
    dpct::err0 status =
        DPCT_CHECK_ERROR(dpct::dpct_free(x_gpu, dpct::get_in_order_queue()));
    check_error(status);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void cuda_push_array(float *x_gpu, float *x, size_t n) try {
    size_t size = sizeof(float)*n;
    dpct::err0 status = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(x_gpu, x, size).wait());
    check_error(status);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n) try {
    size_t size = sizeof(float)*n;
    dpct::err0 status = DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(x, x_gpu, size).wait());
    check_error(status);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = (float *)calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}
#else
void cuda_set_device(int n){}

#endif
