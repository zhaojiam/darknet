#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void INTERCEPT(float *devPtr, char *msgStr, int numOfVar, int offset);

#ifdef __cplusplus
}
#endif