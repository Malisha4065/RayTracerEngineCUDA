#ifndef SCENE_H
#define SCENE_H

#include "hittables.h"
#include "camera_shared.h"
#include <SDL2/SDL.h>

// Host-side scene management
void add_sphere_to_scene_host(Vec3 center, float radius, Material_Device mat);
void add_cube_to_scene_host(Vec3 center, Vec3 size, Material_Device mat);
void init_engine_scene_and_gpu_data();
void render_frame_cuda(SDL_Renderer *renderer, SDL_Texture *texture);
void cleanup_gpu_data();

// Host-side random for scene generation
float host_random_float();
float host_random_float_range(float min, float max);

// Global variables (extern declarations)
extern SphereData_Device h_spheres_data[MAX_OBJECTS];
extern int h_num_spheres;
extern CubeData_Device h_cubes_data[MAX_OBJECTS];
extern int h_num_cubes;

extern SphereData_Device* d_spheres_data;
extern CubeData_Device* d_cubes_data;
extern Camera_Device d_camera;
extern Vec3* d_pixel_data;
extern curandState* d_rand_states;

extern Vec3 g_camera_pos_host;
extern Vec3 g_camera_lookat_host;
extern Vec3 g_camera_vup_host;
extern float g_fov_y_degrees_host;
extern Vec3 g_pivot_point_host;
extern float g_distance_to_pivot_host;
extern float g_camera_yaw_host;
extern float g_camera_pitch_host;

#endif // SCENE_H