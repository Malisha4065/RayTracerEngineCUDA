#ifndef MATERIALS_H
#define MATERIALS_H

#include "common_math.h"
#include "cuda_utils.h"

// --- Material (Device) ---
typedef enum {
    MAT_LAMBERTIAN_DEVICE,
    MAT_METAL_DEVICE,
    MAT_EMISSIVE_DEVICE
} MaterialType_Device;

typedef struct {
    MaterialType_Device type;
    Vec3 albedo;
    Vec3 emission;
    float fuzz;
} Material_Device;

// --- HitRecord (Device) ---
typedef struct {
    Vec3 p;
    Vec3 normal;
    Material_Device material;
    float t;
    bool front_face;
} HitRecord_Device;

__device__ inline void hit_record_set_face_normal_device(HitRecord_Device* rec, const Ray* r, const Vec3* outward_normal) {
    rec->front_face = vec3_dot(r->direction, *outward_normal) < 0.0f;
    rec->normal = rec->front_face ? *outward_normal : vec3_scale(*outward_normal, -1.0f);
}

// Material functions
__device__ bool lambertian_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state);
__device__ bool metal_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state);
__device__ bool emissive_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state);
__device__ bool material_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state);
__device__ Vec3 material_emitted_device(const Material_Device* self, const HitRecord_Device* rec);

// Host-side material creation helpers
Material_Device material_lambertian_create_host(Vec3 albedo);
Material_Device material_metal_create_host(Vec3 albedo, float fuzz);
Material_Device material_emissive_create_host(Vec3 emission_color);

#endif // MATERIALS_H