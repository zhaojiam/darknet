#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void INTERCEPT(float *devPtr, char *msgStr, int numOfVar, int offset);
void sync_current_device();
void set_memory_for_dnnl(
    void **srcTensorDesc, void **dstTensorDesc,   // dpct::dnnl::memory_desc_ext
    void **dsrcTensorDesc, void **ddstTensorDesc, 
    void **normTensorDesc, void **weightDesc,     
    void **dweightDesc,                          
    void **convDesc,                             // dpct::dnnl::convolution_desc
    void **fw_algo, void **bd_algo, void **bf_algo // dnnl::algorithm
    );

#ifdef __cplusplus
}
#endif