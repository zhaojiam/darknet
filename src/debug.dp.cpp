#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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