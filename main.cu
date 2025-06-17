#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <SDL2/SDL.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_DEPTH 5
#define SAMPLES_PER_PIXEL 1
#define MAX_OBJECTS 10

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INFINITY_CUDA FLT_MAX

// CUDA error checking macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// --- Vec3 (Device) ---
typedef struct { float x, y, z; } Vec3;

__host__ __device__ inline Vec3 vec3_create(float x, float y, float z) { return (Vec3){x, y, z}; }
__host__ __device__ inline Vec3 vec3_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ inline Vec3 vec3_sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ inline Vec3 vec3_mul(Vec3 a, Vec3 b) { return (Vec3){a.x * b.x, a.y * b.y, a.z * b.z}; }
__host__ __device__ inline Vec3 vec3_scale(Vec3 a, float s) { return (Vec3){a.x * s, a.y * s, a.z * s}; }
__host__ __device__ inline Vec3 vec3_div(Vec3 a, float s) { float inv_s = 1.0f / s; return vec3_scale(a, inv_s); }
__host__ __device__ inline float vec3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ inline Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
__host__ __device__ inline float vec3_length_squared(Vec3 a) { return vec3_dot(a, a); }
__host__ __device__ inline float vec3_length(Vec3 a) { return sqrtf(vec3_length_squared(a)); }
__host__ __device__ inline Vec3 vec3_normalize(Vec3 a) {
    float len = vec3_length(a);
    if (len == 0.0f) return (Vec3){0,0,0};
    return vec3_div(a, len);
}
__host__ __device__ inline Vec3 vec3_reflect(Vec3 v, Vec3 n) {
    return vec3_sub(v, vec3_scale(n, 2.0f * vec3_dot(v, n)));
}

// --- Ray (Device) ---
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

__host__ __device__ inline Ray ray_create(Vec3 origin, Vec3 direction) {
    Ray r;
    r.origin = origin;
    r.direction = direction; // Normalization happens in camera or scatter
    return r;
}
__host__ __device__ inline Vec3 ray_at(Ray r, float t) {
    return vec3_add(r.origin, vec3_scale(r.direction, t));
}

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
    Material_Device material; // Material is copied
    float t;
    bool front_face;
} HitRecord_Device;

__device__ inline void hit_record_set_face_normal_device(HitRecord_Device* rec, const Ray* r, const Vec3* outward_normal) {
    rec->front_face = vec3_dot(r->direction, *outward_normal) < 0.0f;
    rec->normal = rec->front_face ? *outward_normal : vec3_scale(*outward_normal, -1.0f);
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


// --- Material Scatter/Emitted Logic (Device) ---
__device__ bool lambertian_scatter_device(const Material_Device* self, const Ray* r_in, const HitRecord_Device* rec, Vec3* attenuation, Ray* scattered_ray, curandState* local_rand_state) {
    (void)r_in;
    Vec3 scatter_direction = vec3_add(rec->normal, random_unit_vector_device(local_rand_state));
    if (vec3_length_squared(scatter_direction) < 1e-8f) { // Catch degenerate scatter direction
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
    return false; // Emissive materials don't scatter
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
    (void)rec; // Unused for simple emissive
    if (self->type == MAT_EMISSIVE_DEVICE) {
        return self->emission;
    }
    return vec3_create(0,0,0);
}


// --- Sphere (Device) ---
typedef struct {
    Vec3 center;
    float radius;
    Material_Device material;
} SphereData_Device;

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

// --- Cube (Device) ---
typedef struct {
    Vec3 min_corner;
    Vec3 max_corner;
    Material_Device material;
} CubeData_Device;

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

// --- Camera (Device part) ---
typedef struct {
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    // u,v,w are implicitly used in calculation of lower_left_corner, horizontal, vertical
} Camera_Device;

__device__ inline Ray camera_get_ray_device(const Camera_Device* cam, float s, float t) {
    Vec3 term_s = vec3_scale(cam->horizontal, s);
    Vec3 term_t = vec3_scale(cam->vertical, t);
    Vec3 point_on_plane = vec3_add(cam->lower_left_corner, term_s);
    point_on_plane = vec3_add(point_on_plane, term_t);
    
    Vec3 direction = vec3_sub(point_on_plane, cam->origin);
    return ray_create(cam->origin, vec3_normalize(direction)); // Ensure direction is normalized
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
    curandState local_rand_state = rand_states[pixel_index]; // Each thread gets its own state

    Vec3 pixel_color = vec3_create(0,0,0);
    for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
        float u = (float)(x + ((SAMPLES_PER_PIXEL > 1) ? random_float_device(&local_rand_state) : 0.5f)) / (width - 1);
        // Invert Y for image coordinates vs camera coordinates
        float v_img = (float)(y + ((SAMPLES_PER_PIXEL > 1) ? random_float_device(&local_rand_state) : 0.5f)) / (height - 1);
        float v_cam = 1.0f - v_img; // Or (height - 1 - y_coord_with_offset) / (height - 1)

        Ray r = camera_get_ray_device(&cam, u, v_cam);
        pixel_color = vec3_add(pixel_color, ray_color_device(&r, spheres, num_spheres, cubes, num_cubes, MAX_DEPTH, &local_rand_state));
    }
    pixel_color = vec3_div(pixel_color, (float)SAMPLES_PER_PIXEL);

    // Gamma correction
    pixel_color.x = sqrtf(fmaxf(0.0f, pixel_color.x));
    pixel_color.y = sqrtf(fmaxf(0.0f, pixel_color.y));
    pixel_color.z = sqrtf(fmaxf(0.0f, pixel_color.z));
    
    fb[pixel_index] = pixel_color;
    rand_states[pixel_index] = local_rand_state; // Save state for next frame if needed (not strictly necessary here)
}

// --- Kernel to initialize cuRAND states ---
__global__ void init_random_states_kernel(curandState *rand_states, int num_states, unsigned long long seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        // Initialize each state with a unique seed
        curand_init(seed_offset + idx, 0, 0, &rand_states[idx]);
    }
}


// --- Host-side scene storage and setup ---
// Host-side storage for scene objects before copying to GPU
SphereData_Device h_spheres_data[MAX_OBJECTS];
int h_num_spheres = 0;
CubeData_Device h_cubes_data[MAX_OBJECTS];
int h_num_cubes = 0;

// Device pointers
SphereData_Device* d_spheres_data = NULL;
CubeData_Device* d_cubes_data = NULL;
Camera_Device d_camera;
Vec3* d_pixel_data = NULL; // Framebuffer on GPU
curandState* d_rand_states = NULL;

// Host-side camera parameters (similar to original globals)
Vec3 g_camera_pos_host = {0,0,0};
Vec3 g_camera_lookat_host;
Vec3 g_camera_vup_host = {0,1,0};
float g_fov_y_degrees_host = 60.0f;
Vec3 g_pivot_point_host;
float g_distance_to_pivot_host;
float g_camera_yaw_host = 0.0f;
float g_camera_pitch_host = 0.0f;


// Host-side Material creation helpers
Material_Device material_lambertian_create_host(Vec3 albedo) {
    Material_Device mat;
    mat.type = MAT_LAMBERTIAN_DEVICE;
    mat.albedo = albedo;
    mat.emission = vec3_create(0,0,0);
    mat.fuzz = 0;
    return mat;
}

Material_Device material_metal_create_host(Vec3 albedo, float fuzz) {
    Material_Device mat;
    mat.type = MAT_METAL_DEVICE;
    mat.albedo = albedo;
    mat.emission = vec3_create(0,0,0);
    mat.fuzz = fuzz < 1.0f ? fuzz : 1.0f;
    return mat;
}

Material_Device material_emissive_create_host(Vec3 emission_color) {
    Material_Device mat;
    mat.type = MAT_EMISSIVE_DEVICE;
    mat.albedo = vec3_create(0,0,0);
    mat.emission = emission_color;
    mat.fuzz = 0;
    return mat;
}

// Host-side random for scene generation
float host_random_float() { return (float)rand() / (RAND_MAX + 1.0f); }
float host_random_float_range(float min, float max) { return min + (max-min)*host_random_float(); }


void add_sphere_to_scene_host(Vec3 center, float radius, Material_Device mat) {
    if (h_num_spheres < MAX_OBJECTS) {
        h_spheres_data[h_num_spheres].center = center;
        h_spheres_data[h_num_spheres].radius = radius;
        h_spheres_data[h_num_spheres].material = mat;
        h_num_spheres++;
    } else {
        fprintf(stderr, "Max sphere objects reached for host storage.\n");
    }
}

void add_cube_to_scene_host(Vec3 center, Vec3 size, Material_Device mat) {
    if (h_num_cubes < MAX_OBJECTS) {
        Vec3 half_size = vec3_scale(size, 0.5f);
        h_cubes_data[h_num_cubes].min_corner = vec3_sub(center, half_size);
        h_cubes_data[h_num_cubes].max_corner = vec3_add(center, half_size);
        h_cubes_data[h_num_cubes].material = mat;
        h_num_cubes++;
    } else {
        fprintf(stderr, "Max cube objects reached for host storage.\n");
    }
}

// Host-side camera initialization (calculates parameters for Camera_Device)
void camera_init_host(Camera_Device* cam_device_params, Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov_degrees, float aspect_ratio) {
    float theta = vfov_degrees * M_PI / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    Vec3 w = vec3_normalize(vec3_sub(lookfrom, lookat));
    Vec3 u = vec3_normalize(vec3_cross(vup, w));
    Vec3 v = vec3_cross(w, u);

    cam_device_params->origin = lookfrom;
    cam_device_params->horizontal = vec3_scale(u, viewport_width);
    cam_device_params->vertical = vec3_scale(v, viewport_height);
    
    Vec3 term1 = vec3_div(cam_device_params->horizontal, 2.0f);
    Vec3 term2 = vec3_div(cam_device_params->vertical, 2.0f);
    // focal_length is 1.0, w is unit vector along view direction (lookfrom - lookat)
    // lower_left_corner = origin - horizontal/2 - vertical/2 - w
    cam_device_params->lower_left_corner = vec3_sub(cam_device_params->origin, term1);
    cam_device_params->lower_left_corner = vec3_sub(cam_device_params->lower_left_corner, term2);
    cam_device_params->lower_left_corner = vec3_sub(cam_device_params->lower_left_corner, w);
}


void init_engine_scene_and_gpu_data() {
    h_num_spheres = 0;
    h_num_cubes = 0;

    g_pivot_point_host = vec3_create(0.0f, 0.0f, -4.0f); 
    bool pivot_set_by_light = false;

    int num_total_objects;
    printf("Enter the total number of objects for the scene (0 to %d): ", MAX_OBJECTS);
    scanf("%d", &num_total_objects);
    int c;
    while ((c = getchar()) != '\n' && c != EOF); // Clear input buffer

    if (num_total_objects < 0) num_total_objects = 0;
    if (num_total_objects > MAX_OBJECTS) {
        printf("Warning: Number of objects exceeds MAX_OBJECTS (%d). Clamping to MAX_OBJECTS.\n", MAX_OBJECTS);
        num_total_objects = MAX_OBJECTS;
    }
    
    for (int i = 0; i < num_total_objects; ++i) {
        printf("\nConfiguring Object %d of %d:\n", i + 1, num_total_objects);
        int shape_type_choice;
        printf("  Select shape type (1: Sphere, 2: Cube): ");
        scanf("%d", &shape_type_choice);
        while ((c = getchar()) != '\n' && c != EOF);

        Vec3 center;
        Material_Device mat;
        int mat_type_choice;

        printf("  Select material type for object %d (1: Diffuse, 2: Metal, 3: Light): ", i + 1);
        scanf("%d", &mat_type_choice);
        while ((c = getchar()) != '\n' && c != EOF);

        Vec3 random_color_val = vec3_create(host_random_float(), host_random_float(), host_random_float());
        random_color_val.x = fmaxf(0.1f, random_color_val.x); // Ensure not too dark
        random_color_val.y = fmaxf(0.1f, random_color_val.y);
        random_color_val.z = fmaxf(0.1f, random_color_val.z);

        switch (mat_type_choice) {
            case 1: mat = material_lambertian_create_host(random_color_val); break;
            case 2: mat = material_metal_create_host(random_color_val, host_random_float_range(0.0f, 0.4f)); break;
            case 3: {
                Vec3 emission_color = vec3_scale(random_color_val, host_random_float_range(1.5f, 4.0f));
                mat = material_emissive_create_host(emission_color);
                break;
            }
            default:
                printf("    Invalid material choice. Defaulting to Lambertian grey.\n");
                mat = material_lambertian_create_host(vec3_create(0.5f, 0.5f, 0.5f));
                break;
        }

        float grid_spacing = 2.0f;
        int items_per_row = 3; 
        float x_pos = (i % items_per_row - (items_per_row -1) / 2.0f) * grid_spacing + host_random_float_range(-0.3f, 0.3f);
        float y_pos = host_random_float_range(-1.0f, 1.0f); 
        float z_pos = -5.0f - (i / items_per_row) * grid_spacing + host_random_float_range(-0.5f, 0.5f);
        center = vec3_create(x_pos, y_pos, z_pos);

        if (shape_type_choice == 1) { // Sphere
            float radius = host_random_float_range(0.3f, 0.7f);
            add_sphere_to_scene_host(center, radius, mat);
            printf("    -> Created Sphere at (%.2f, %.2f, %.2f), radius %.2f\n", center.x, center.y, center.z, radius);
        } else if (shape_type_choice == 2) { // Cube
            Vec3 size = vec3_create(host_random_float_range(0.5f, 1.2f), host_random_float_range(0.5f, 1.2f), host_random_float_range(0.5f, 1.2f));
            add_cube_to_scene_host(center, size, mat);
            printf("    -> Created Cube centered at (%.2f, %.2f, %.2f) with size (%.2f, %.2f, %.2f)\n", center.x, center.y, center.z, size.x, size.y, size.z);
        } else {
            printf("    Invalid shape choice. Skipping object.\n");
            continue;
        }
        if (mat_type_choice == 3 && !pivot_set_by_light) {
            g_pivot_point_host = center;
            pivot_set_by_light = true;
            printf("    -> This Light Source object will be the pivot point.\n");
        }
    }
    
    g_camera_lookat_host = g_pivot_point_host;
    g_distance_to_pivot_host = vec3_length(vec3_sub(g_camera_pos_host, g_pivot_point_host));
    if (g_distance_to_pivot_host < 0.001f) {
        g_camera_pos_host.z = g_pivot_point_host.z - 5.0f;
        g_distance_to_pivot_host = 5.0f;
    }
    Vec3 initial_view_dir = vec3_normalize(vec3_sub(g_camera_lookat_host, g_camera_pos_host));
    g_camera_pitch_host = asinf(initial_view_dir.y);
    g_camera_yaw_host = atan2f(initial_view_dir.x, -initial_view_dir.z);

    printf("\nInitializing camera with default settings:\n");
    printf("  Position: (%.1f, %.1f, %.1f)\n", g_camera_pos_host.x, g_camera_pos_host.y, g_camera_pos_host.z);
    printf("  Looking at (Pivot): (%.1f, %.1f, %.1f)\n", g_camera_lookat_host.x, g_camera_lookat_host.y, g_camera_lookat_host.z);

    // Allocate GPU memory
    if (h_num_spheres > 0) {
        gpuErrchk(cudaMalloc((void**)&d_spheres_data, h_num_spheres * sizeof(SphereData_Device)));
        gpuErrchk(cudaMemcpy(d_spheres_data, h_spheres_data, h_num_spheres * sizeof(SphereData_Device), cudaMemcpyHostToDevice));
    }
    if (h_num_cubes > 0) {
        gpuErrchk(cudaMalloc((void**)&d_cubes_data, h_num_cubes * sizeof(CubeData_Device)));
        gpuErrchk(cudaMemcpy(d_cubes_data, h_cubes_data, h_num_cubes * sizeof(CubeData_Device), cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMalloc((void**)&d_pixel_data, WIDTH * HEIGHT * sizeof(Vec3)));
    gpuErrchk(cudaMalloc((void**)&d_rand_states, WIDTH * HEIGHT * sizeof(curandState)));

    // Initialize random states on GPU using the new kernel
    int num_rand_states = WIDTH * HEIGHT;
    dim3 threadsPerBlockRand(256); // Can be 1D for this kernel
    dim3 numBlocksRand((num_rand_states + threadsPerBlockRand.x - 1) / threadsPerBlockRand.x);
    init_random_states_kernel<<<numBlocksRand, threadsPerBlockRand>>>(d_rand_states, num_rand_states, (unsigned long long)time(NULL));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize()); // Ensure kernel finishes before proceeding

    // Remove old host-side initialization of rand_states
    // curandState *h_rand_states = (curandState*)malloc(WIDTH * HEIGHT * sizeof(curandState));
    // for(int i = 0; i < WIDTH * HEIGHT; ++i) {
    //     curand_init(time(NULL) + i, 0, 0, &h_rand_states[i]); // Seed with time + index
    // }
    // gpuErrchk(cudaMemcpy(d_rand_states, h_rand_states, WIDTH * HEIGHT * sizeof(curandState), cudaMemcpyHostToDevice));
    // free(h_rand_states);
}

void render_frame_cuda(SDL_Renderer *renderer, SDL_Texture *texture) {
    // Update camera parameters for d_camera
    camera_init_host(&d_camera, g_camera_pos_host, g_camera_lookat_host, g_camera_vup_host, g_fov_y_degrees_host, (float)WIDTH / HEIGHT);

    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_kernel<<<numBlocks, threadsPerBlock>>>(d_pixel_data, WIDTH, HEIGHT, d_camera,
                                                 d_spheres_data, h_num_spheres,
                                                 d_cubes_data, h_num_cubes,
                                                 d_rand_states);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy result back to host
    Vec3* h_pixels = (Vec3*)malloc(WIDTH * HEIGHT * sizeof(Vec3));
    gpuErrchk(cudaMemcpy(h_pixels, d_pixel_data, WIDTH * HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost));

    // Update SDL Texture
    void *sdl_pixels_locked;
    int pitch;
    SDL_LockTexture(texture, NULL, &sdl_pixels_locked, &pitch);
    unsigned char *pixel_data_sdl = (unsigned char *)sdl_pixels_locked;

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            Vec3 p_color = h_pixels[y * WIDTH + x];
            int ir = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.x)));
            int ig = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.y)));
            int ib = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.z)));

            int index = y * pitch + x * 4; // Assuming ARGB8888 or RGBA8888
            // SDL_PIXELFORMAT_ARGB8888 means B,G,R,A in memory on little-endian like x86
            // Check your SDL_PIXELFORMAT if colors are swapped
            pixel_data_sdl[index + 0] = (unsigned char)ib; // Blue
            pixel_data_sdl[index + 1] = (unsigned char)ig; // Green
            pixel_data_sdl[index + 2] = (unsigned char)ir; // Red
            pixel_data_sdl[index + 3] = 255;               // Alpha
        }
    }
    free(h_pixels);
    SDL_UnlockTexture(texture);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

void cleanup_gpu_data() {
    if (d_spheres_data) gpuErrchk(cudaFree(d_spheres_data));
    if (d_cubes_data) gpuErrchk(cudaFree(d_cubes_data));
    if (d_pixel_data) gpuErrchk(cudaFree(d_pixel_data));
    if (d_rand_states) gpuErrchk(cudaFree(d_rand_states));
    d_spheres_data = NULL;
    d_cubes_data = NULL;
    d_pixel_data = NULL;
    d_rand_states = NULL;
}


int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));

    gpuErrchk(cudaFree(0)); 
    size_t new_stack_size = 16384; // Example: 16 KB, adjust as needed
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));

    size_t current_stack_size;
    gpuErrchk(cudaDeviceGetLimit(&current_stack_size, cudaLimitStackSize));
    printf("CUDA device stack size set to: %zu bytes\n", current_stack_size);

    init_engine_scene_and_gpu_data();

    if (SDL_Init(SDL_INIT_VIDEO) < 0) { 
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError()); return 1; 
    }
    SDL_Window *window = SDL_CreateWindow("Raytracer Engine CUDA", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) { 
        fprintf(stderr, "SDL_CreateWindow Error: %s\n", SDL_GetError()); SDL_Quit(); return 1; 
    }
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) { 
        fprintf(stderr, "SDL_CreateRenderer Error: %s\n", SDL_GetError()); SDL_DestroyWindow(window); SDL_Quit(); return 1; 
    }
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    if (!texture) { 
        fprintf(stderr, "SDL_CreateTexture Error: %s\n", SDL_GetError()); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit(); return 1; 
    }

    Uint32 startTime, endTime;
    SDL_Event e;
    int quit = 0;
    int mouse_down = 0;
    int needs_render = 1;
    static int is_fullscreen = 0;

    const float key_rotate_speed = 0.05f;
    const float key_zoom_speed = 0.25f;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) quit = 1;
            else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                mouse_down = 1; SDL_SetRelativeMouseMode(SDL_TRUE);
            } else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                mouse_down = 0; SDL_SetRelativeMouseMode(SDL_FALSE);
            } else if (e.type == SDL_MOUSEMOTION && mouse_down) {
                float sensitivity = 0.0025f;
                g_camera_yaw_host += (float)e.motion.xrel * sensitivity;
                g_camera_pitch_host -= (float)e.motion.yrel * sensitivity;
                const float pitch_limit = (M_PI / 2.0f) - 0.01f;
                if (g_camera_pitch_host > pitch_limit) g_camera_pitch_host = pitch_limit;
                if (g_camera_pitch_host < -pitch_limit) g_camera_pitch_host = -pitch_limit;
                needs_render = 1;
            } else if (e.type == SDL_KEYDOWN) {
                int key_action_taken = 0;
                switch (e.key.keysym.sym) {
                    case SDLK_f: is_fullscreen = !is_fullscreen; SDL_SetWindowFullscreen(window, is_fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0); key_action_taken = 1; break;
                    case SDLK_LEFT: case SDLK_a: g_camera_yaw_host -= key_rotate_speed; key_action_taken = 1; break;
                    case SDLK_RIGHT: case SDLK_d: g_camera_yaw_host += key_rotate_speed; key_action_taken = 1; break;
                    case SDLK_UP: case SDLK_w: g_camera_pitch_host += key_rotate_speed; key_action_taken = 1; break;
                    case SDLK_DOWN: case SDLK_s: g_camera_pitch_host -= key_rotate_speed; key_action_taken = 1; break;
                    case SDLK_PLUS: case SDLK_EQUALS: case SDLK_KP_PLUS: g_distance_to_pivot_host -= key_zoom_speed; key_action_taken = 1; break;
                    case SDLK_MINUS: case SDLK_KP_MINUS: g_distance_to_pivot_host += key_zoom_speed; key_action_taken = 1; break;
                }
                if (key_action_taken) {
                    const float pitch_limit = (M_PI / 2.0f) - 0.01f;
                    if (g_camera_pitch_host > pitch_limit) g_camera_pitch_host = pitch_limit;
                    if (g_camera_pitch_host < -pitch_limit) g_camera_pitch_host = -pitch_limit;
                    g_distance_to_pivot_host = fmaxf(0.5f, g_distance_to_pivot_host);
                    needs_render = 1;
                }
            } else if (e.type == SDL_MOUSEWHEEL) {
                float distance_zoom_speed = 0.5f;
                if (e.wheel.y > 0) g_distance_to_pivot_host -= distance_zoom_speed;
                else if (e.wheel.y < 0) g_distance_to_pivot_host += distance_zoom_speed;
                g_distance_to_pivot_host = fmaxf(0.5f, g_distance_to_pivot_host);
                needs_render = 1;
            } else if (e.type == SDL_MULTIGESTURE) {
                 if (e.mgesture.numFingers >= 2) { 
                    float touchpad_zoom_sensitivity = 5.0f; 
                    g_distance_to_pivot_host += e.mgesture.dDist * touchpad_zoom_sensitivity;
                    g_distance_to_pivot_host = fmaxf(0.5f, g_distance_to_pivot_host); 
                    needs_render = 1; 
                }
            }
        }

        if (needs_render) {
            float cam_offset_x = g_distance_to_pivot_host * cosf(g_camera_pitch_host) * sinf(g_camera_yaw_host);
            float cam_offset_y = g_distance_to_pivot_host * sinf(g_camera_pitch_host);
            float cam_offset_z = g_distance_to_pivot_host * cosf(g_camera_pitch_host) * -cosf(g_camera_yaw_host); // Negative cos for Z
            g_camera_pos_host.x = g_pivot_point_host.x + cam_offset_x;
            g_camera_pos_host.y = g_pivot_point_host.y + cam_offset_y;
            g_camera_pos_host.z = g_pivot_point_host.z + cam_offset_z;
        }

        if (needs_render) {
            startTime = SDL_GetTicks();
            render_frame_cuda(renderer, texture);
            endTime = SDL_GetTicks();
            printf("Render time: %u ms\n", endTime - startTime);
            needs_render = 0;
        }
        // SDL_Delay(1); // a small delay if CPU usage is too high in idle
    }

    cleanup_gpu_data();
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
