#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void INTERCEPT(float *devPtr, char *msgStr, int numOfVar, int offset);
void sync_current_device();

#ifdef __cplusplus
}
#endif