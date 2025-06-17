#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- Random numbers (Device) ---
__device__ inline float random_float_device(curandState *local_rand_state) {
    return curand_uniform(local_rand_state);
}

__device__ inline float random_float_range_device(curandState *local_rand_state, float min, float max) {
    return min + (max - min) * random_float_device(local_rand_state);
}

__device__ inline Vec3 random_in_unit_sphere_device(curandState *local_rand_state) {
    while (true) {
        Vec3 p = vec3_create(random_float_range_device(local_rand_state, -1.0f, 1.0f),
                             random_float_range_device(local_rand_state, -1.0f, 1.0f),
                             random_float_range_device(local_rand_state, -1.0f, 1.0f));
        if (vec3_length_squared(p) < 1.0f) return p;
    }
}

__device__ inline Vec3 random_unit_vector_device(curandState *local_rand_state) {
    return vec3_normalize(random_in_unit_sphere_device(local_rand_state));
}

#endif // CUDA_UTILS_H