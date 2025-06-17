#ifndef CAMERA_SHARED_H
#define CAMERA_SHARED_H

#include "common_math.h"

// --- Camera (Device part) ---
typedef struct {
    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
} Camera_Device;

__device__ inline Ray camera_get_ray_device(const Camera_Device* cam, float s, float t) {
    Vec3 term_s = vec3_scale(cam->horizontal, s);
    Vec3 term_t = vec3_scale(cam->vertical, t);
    Vec3 point_on_plane = vec3_add(cam->lower_left_corner, term_s);
    point_on_plane = vec3_add(point_on_plane, term_t);
    
    Vec3 direction = vec3_sub(point_on_plane, cam->origin);
    return ray_create(cam->origin, vec3_normalize(direction));
}

// Host-side camera initialization
void camera_init_host(Camera_Device* cam_device_params, Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov_degrees, float aspect_ratio);

#endif // CAMERA_SHARED_H