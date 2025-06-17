#ifndef HITTABLES_H
#define HITTABLES_H

#include "common_math.h"
#include "materials.h"

// --- Sphere (Device) ---
typedef struct {
    Vec3 center;
    float radius;
    Material_Device material;
} SphereData_Device;

__device__ bool sphere_hit_device(const SphereData_Device* sphere, const Ray* r, float t_min, float t_max, HitRecord_Device* rec);

// --- Cube (Device) ---
typedef struct {
    Vec3 min_corner;
    Vec3 max_corner;
    Material_Device material;
} CubeData_Device;

__device__ bool cube_hit_device(const CubeData_Device* cube, const Ray* r, float t_min, float t_max, HitRecord_Device* rec);

// --- World Hit (Device) ---
__device__ bool world_hit_device(const SphereData_Device* spheres, int num_spheres,
                                 const CubeData_Device* cubes, int num_cubes,
                                 const Ray* r, float t_min, float t_max, HitRecord_Device* rec);

#endif // HITTABLES_H