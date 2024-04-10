#include <iostream>
#include "debug.h"

#define DEBUG_ARRAY_SIZE 100
float debugArray[DEBUG_ARRAY_SIZE];
void INTERCEPT(float *devPtr, char *msgStr, int numOfVar, int offset)             
{                                                               
    for (int i = 0; i < DEBUG_ARRAY_SIZE; i++) debugArray[i]=0; 
    printf("----------------------\n");                         
    cudaDeviceSynchronize();                                    
    cudaMemcpy(debugArray, (devPtr)+(offset), sizeof(float)*(numOfVar), cudaMemcpyDeviceToHost);  
    cudaDeviceSynchronize();                                    
    printf("%s: \n", (msgStr));                                 
    for(int i=0; i<(numOfVar);i++) printf("%e ", (float)debugArray[i]); 
    printf("\n----------------------\n");                       
}                                                               

