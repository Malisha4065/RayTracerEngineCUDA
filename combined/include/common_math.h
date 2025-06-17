#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define WIDTH 800
#define HEIGHT 600
#define MAX_DEPTH 5
#define SAMPLES_PER_PIXEL 1
#define MAX_OBJECTS 10

#define INFINITY_CUDA FLT_MAX

// --- Vec3 ---
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

// --- Ray ---
typedef struct {
    Vec3 origin;
    Vec3 direction;
} Ray;

__host__ __device__ inline Ray ray_create(Vec3 origin, Vec3 direction) {
    Ray r;
    r.origin = origin;
    r.direction = direction;
    return r;
}
__host__ __device__ inline Vec3 ray_at(Ray r, float t) {
    return vec3_add(r.origin, vec3_scale(r.direction, t));
}

#endif // COMMON_MATH_H