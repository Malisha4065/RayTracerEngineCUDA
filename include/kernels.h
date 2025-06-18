#ifndef KERNELS_H
#define KERNELS_H

#include "common_math.h"
#include "hittables.h"
#include "camera_shared.h"
#include <curand_kernel.h>

// --- Ray Color (Device) ---
__device__ Vec3 ray_color_device(const Ray* r,
                                 const SphereData_Device* spheres, int num_spheres,
                                 const CubeData_Device* cubes, int num_cubes,
                                 int depth, curandState *local_rand_state);

// --- Kernels ---
__global__ void render_kernel(Vec3* fb, int width, int height,
                              Camera_Device cam,
                              SphereData_Device* spheres, int num_spheres,
                              CubeData_Device* cubes, int num_cubes,
                              curandState *rand_states);

__global__ void init_random_states_kernel(curandState *rand_states, int num_states, unsigned long long seed_offset);

#endif // KERNELS_H