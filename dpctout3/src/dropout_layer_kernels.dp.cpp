#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/rng_utils.hpp>
#include <dpct/blas_utils.hpp>

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

void yoloswag420blazeit360noscope(float *input, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);
    /*
    int i;
    for(i = 0; i < size; ++i){
        layer.rand[i] = rand_uniform();
    }
    cuda_push_array(layer.rand_gpu, layer.rand, size);
    */

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    /*
    DPCT1010:59: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>(net.delta_gpu, size, layer.rand_gpu, layer.probability, layer.scale);
    /*
    DPCT1010:60: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    check_error(0);
}
