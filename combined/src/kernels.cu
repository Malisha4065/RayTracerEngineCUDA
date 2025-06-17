#include <float.h>
#include "../include/kernels.h"
#include "../include/materials.h"
#include "../include/hittables.h"
#include "../include/cuda_utils.h"

// --- Material Scatter/Emitted Logic (Device) ---
__device__ bool lambertian_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state) {
    (void)r_in;
    Vec3 scatter_direction = vec3_add(rec->normal, random_unit_vector_device(local_rand_state));
    if (vec3_length_squared(scatter_direction) < 1e-8f) {
        scatter_direction = rec->normal;
    }
    *scattered_ray = ray_create(rec->p, vec3_normalize(scatter_direction));
    *attenuation = self->albedo;
    return true;
}

__device__ bool metal_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state) {
    Vec3 reflected_dir = vec3_reflect(vec3_normalize(r_in->direction), rec->normal);
    Vec3 fuzzed_dir = vec3_add(reflected_dir, vec3_scale(random_in_unit_sphere_device(local_rand_state), self->fuzz));
    *scattered_ray = ray_create(rec->p, vec3_normalize(fuzzed_dir));
    *attenuation = self->albedo;
    return (vec3_dot(scattered_ray->direction, rec->normal) > 0.0f);
}

__device__ bool emissive_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state) {
    (void)self; (void)r_in; (void)rec; (void)attenuation; (void)scattered_ray; (void)local_rand_state;
    return false;
}

__device__ bool material_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state) {
    switch (self->type) {
        case MAT_LAMBERTIAN_DEVICE:
            return lambertian_scatter_device(self, r_in, rec, attenuation, scattered_ray, local_rand_state);
        case MAT_METAL_DEVICE:
            return metal_scatter_device(self, r_in, rec, attenuation, scattered_ray, local_rand_state);
        case MAT_EMISSIVE_DEVICE:
            return emissive_scatter_device(self, r_in, rec, attenuation, scattered_ray, local_rand_state);
        default:
            return false;
    }
}

__device__ Vec3 material_emitted_device(const Material_Device* self, const HitRecord_Device* rec) {
    (void)rec;
    if (self->type == MAT_EMISSIVE_DEVICE) {
        return self->emission;
    }
    return vec3_create(0,0,0);
}

// --- Sphere Hit (Device) ---
__device__ bool sphere_hit_device(const SphereData_Device* sphere, const Ray* r, float t_min, float t_max, HitRecord_Device* rec) {
    Vec3 oc = vec3_sub(r->origin, sphere->center);
    float a = vec3_length_squared(r->direction);
    float half_b = vec3_dot(oc, r->direction);
    float c = vec3_length_squared(oc) - sphere->radius * sphere->radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0f) return false;
    
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec->t = root;
    rec->p = ray_at(*r, rec->t);
    Vec3 outward_normal = vec3_div(vec3_sub(rec->p, sphere->center), sphere->radius);
    hit_record_set_face_normal_device(rec, r, &outward_normal);
    rec->material = sphere->material;
    return true;
}

// --- Cube Hit (Device) ---
__device__ bool cube_hit_device(const CubeData_Device* cube, const Ray* r, float t_min, float t_max, HitRecord_Device* rec) {
    Vec3 inv_dir = vec3_create(1.0f / r->direction.x, 1.0f / r->direction.y, 1.0f / r->direction.z);

    float t0x = (cube->min_corner.x - r->origin.x) * inv_dir.x;
    float t1x = (cube->max_corner.x - r->origin.x) * inv_dir.x;
    if (inv_dir.x < 0.0f) { float temp = t0x; t0x = t1x; t1x = temp; }

    float t0y = (cube->min_corner.y - r->origin.y) * inv_dir.y;
    float t1y = (cube->max_corner.y - r->origin.y) * inv_dir.y;
    if (inv_dir.y < 0.0f) { float temp = t0y; t0y = t1y; t1y = temp; }

    float t0z = (cube->min_corner.z - r->origin.z) * inv_dir.z;
    float t1z = (cube->max_corner.z - r->origin.z) * inv_dir.z;
    if (inv_dir.z < 0.0f) { float temp = t0z; t0z = t1z; t1z = temp; }

    float t_enter = fmaxf(fmaxf(t0x, t0y), t0z);
    float t_exit = fminf(fminf(t1x, t1y), t1z);

    if (t_enter >= t_exit || t_exit < t_min || t_enter > t_max) {
        return false;
    }

    float t_hit = t_enter;
    if (t_hit < t_min) {
        t_hit = t_exit;
        if (t_hit < t_min || t_hit > t_max) return false;
    }

    rec->t = t_hit;
    rec->p = ray_at(*r, rec->t);
    rec->material = cube->material;

    Vec3 center = vec3_scale(vec3_add(cube->min_corner, cube->max_corner), 0.5f);
    Vec3 p_relative = vec3_sub(rec->p, center);
    Vec3 d = vec3_scale(vec3_sub(cube->max_corner, cube->min_corner), 0.5f);
    float bias = 1.00001f;
    Vec3 outward_normal = {0,0,0};

    if (fabsf(p_relative.x / d.x) * bias >= fabsf(p_relative.y / d.y) && fabsf(p_relative.x / d.x) * bias >= fabsf(p_relative.z / d.z)) {
        outward_normal = vec3_create(p_relative.x > 0 ? 1.0f : -1.0f, 0, 0);
    } else if (fabsf(p_relative.y / d.y) * bias >= fabsf(p_relative.x / d.x) && fabsf(p_relative.y / d.y) * bias >= fabsf(p_relative.z / d.z)) {
        outward_normal = vec3_create(0, p_relative.y > 0 ? 1.0f : -1.0f, 0);
    } else {
        outward_normal = vec3_create(0, 0, p_relative.z > 0 ? 1.0f : -1.0f);
    }
    hit_record_set_face_normal_device(rec, r, &outward_normal);
    return true;
}

// --- World Hit (Device) ---
__device__ bool world_hit_device(const SphereData_Device* spheres, int num_spheres,
                                 const CubeData_Device* cubes, int num_cubes,
                                 const Ray* r, float t_min, float t_max, HitRecord_Device* rec) {
    HitRecord_Device temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < num_spheres; ++i) {
        if (sphere_hit_device(&spheres[i], r, t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *rec = temp_rec;
        }
    }
    for (int i = 0; i < num_cubes; ++i) {
        if (cube_hit_device(&cubes[i], r, t_min, closest_so_far, &temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            *rec = temp_rec;
        }
    }
    return hit_anything;
}

// --- Ray Color (Device) ---
__device__ Vec3 ray_color_device(const Ray* r,
                                 const SphereData_Device* spheres, int num_spheres,
                                 const CubeData_Device* cubes, int num_cubes,
                                 int depth, curandState *local_rand_state) {
    HitRecord_Device rec;

    if (depth <= 0) {
        return vec3_create(0, 0, 0);
    }

    if (world_hit_device(spheres, num_spheres, cubes, num_cubes, r, 0.001f, INFINITY_CUDA, &rec)) {
        Ray scattered_ray;
        Vec3 attenuation;
        Vec3 emitted_light = material_emitted_device(&rec.material, &rec);

        if (material_scatter_device(&rec.material, r, &rec, &attenuation, &scattered_ray, local_rand_state)) {
            Vec3 scattered_color = ray_color_device(&scattered_ray, spheres, num_spheres, cubes, num_cubes, depth - 1, local_rand_state);
            Vec3 final_color;
            final_color.x = emitted_light.x + attenuation.x * scattered_color.x;
            final_color.y = emitted_light.y + attenuation.y * scattered_color.y;
            final_color.z = emitted_light.z + attenuation.z * scattered_color.z;
            return final_color;
        } else {
            return emitted_light;
        }
    }

    // Background
    Vec3 unit_direction = vec3_normalize(r->direction);
    float t = 0.5f * (unit_direction.y + 1.0f);
    Vec3 white = vec3_create(1.0f, 1.0f, 1.0f);
    Vec3 blue = vec3_create(0.5f, 0.7f, 1.0f);
    return vec3_add(vec3_scale(white, 1.0f - t), vec3_scale(blue, t));
}

// --- Render Kernel ---
__global__ void render_kernel(Vec3* fb, int width, int height,
                              Camera_Device cam,
                              SphereData_Device* spheres, int num_spheres,
                              CubeData_Device* cubes, int num_cubes,
                              curandState *rand_states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height || x < 0 || y < 0) return;

    int pixel_index = y * width + x;
    if (pixel_index >= width * height) return;
    curandState local_rand_state = rand_states[pixel_index];

    Vec3 pixel_color = vec3_create(0,0,0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
        float u = (float)(x + ((SAMPLES_PER_PIXEL > 1) ? random_float_device(&local_rand_state) : 0.5f)) / (width - 1);
        float v_img = (float)(y + ((SAMPLES_PER_PIXEL > 1) ? random_float_device(&local_rand_state) : 0.5f)) / (height - 1);
        float v_cam = 1.0f - v_img;

        Ray r = camera_get_ray_device(&cam, u, v_cam);
        pixel_color = vec3_add(pixel_color, ray_color_device(&r, spheres, num_spheres, cubes, num_cubes, MAX_DEPTH, &local_rand_state));
    }
    pixel_color = vec3_div(pixel_color, (float)SAMPLES_PER_PIXEL);

    // Gamma correction
    pixel_color.x = sqrtf(fmaxf(0.0f, pixel_color.x));
    pixel_color.y = sqrtf(fmaxf(0.0f, pixel_color.y));
    pixel_color.z = sqrtf(fmaxf(0.0f, pixel_color.z));
    
    fb[pixel_index] = pixel_color;
    rand_states[pixel_index] = local_rand_state;
}

// --- Kernel to initialize cuRAND states ---
__global__ void init_random_states_kernel(curandState *rand_states, int num_states, unsigned long long seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed_offset + idx, 0, 0, &rand_states[idx]);
    }
}