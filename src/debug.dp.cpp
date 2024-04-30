#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dnnl_utils.hpp>
#include "debug.h"

#define DEBUG_ARRAY_SIZE 100
float debugArray[DEBUG_ARRAY_SIZE];

void INTERCEPT(float *devPtr, char *msgStr, int numOfVar, int offset) 
{                                                               
    sycl::queue q = dpct::get_in_order_queue();                 
    for (int i = 0; i < DEBUG_ARRAY_SIZE; i++) debugArray[i]=0; 
    printf("----------------------\n");                         
    q.wait();                                                   
    q.memcpy(debugArray, (devPtr)+(offset), sizeof(float)*(numOfVar));  
    q.wait();                                                   
    printf("%s: \n", (msgStr));                                 
    for(int i=0; i<(numOfVar);i++) printf("%e ", (float)debugArray[i]); 
    printf("\n----------------------\n");                       
}                                                               


void sync_current_device() {
    dpct::get_current_device().queues_wait_and_throw();
}

void set_memory_for_dnnl(
    void **srcTensorDesc, void **dstTensorDesc,   // dpct::dnnl::memory_desc_ext
    void **dsrcTensorDesc, void **ddstTensorDesc, 
    void **normTensorDesc, void **weightDesc,     
    void **dweightDesc,                          
    void **convDesc,                             // dpct::dnnl::convolution_desc
    void **fw_algo, void **bd_algo, void **bf_algo // dnnl::algorithm
    ) {
    
    *srcTensorDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));
    *dstTensorDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));
    *dsrcTensorDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));
    *ddstTensorDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));
    *normTensorDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));
    *weightDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));
    *dweightDesc = malloc(sizeof(dpct::dnnl::memory_desc_ext));

    *convDesc = malloc(sizeof(dpct::dnnl::convolution_desc));

    *fw_algo = malloc(sizeof(dnnl::algorithm));
    *bd_algo = malloc(sizeof(dnnl::algorithm));
    *bf_algo = malloc(sizeof(dnnl::algorithm));
}