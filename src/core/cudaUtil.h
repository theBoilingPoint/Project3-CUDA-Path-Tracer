#pragma once

#include <cstring>

// Shared CUDA error-check helper. The function is defined in pathtrace.cu
// (guarded by ERRORCHECK); this header lets any translation unit call
// checkCUDAError(msg) without redefining the machinery.
void checkCUDAErrorFn(const char *msg, const char *file, int line);

#define FILENAME                                                               \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
