#include "../include/scene.h"
#include "../include/kernels.h"
#include "../include/cuda_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Global variables definitions
SphereData_Device h_spheres_data[MAX_OBJECTS];
int h_num_spheres = 0;
CubeData_Device h_cubes_data[MAX_OBJECTS];
int h_num_cubes = 0;

// Dynamic resolution variables
int g_current_width = DEFAULT_WIDTH;
int g_current_height = DEFAULT_HEIGHT;

SphereData_Device* d_spheres_data = NULL;
CubeData_Device* d_cubes_data = NULL;
Camera_Device d_camera;
Vec3* d_pixel_data = NULL;
curandState* d_rand_states = NULL;

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
    cam_device_params->lower_left_corner = vec3_sub(cam_device_params->origin, term1);
    cam_device_params->lower_left_corner = vec3_sub(cam_device_params->lower_left_corner, term2);
    cam_device_params->lower_left_corner = vec3_sub(cam_device_params->lower_left_corner, w);
}

void init_engine_scene_and_gpu_data() {
    h_num_spheres = 0;
    h_num_cubes = 0;

    g_pivot_point_host = vec3_create(0.0f, 0.0f, -4.0f); 
    bool pivot_set_by_light = false;

    int scene_choice;
    printf("Select a scene to render:\n");
    printf("  1: Custom scene with user-defined objects\n");
    printf("  2: Pre-defined natural scene (multiple trees and ground)\n");
    printf("Enter your choice (1 or 2): ");
    scanf("%d", &scene_choice);
    int c;
    while ((c = getchar()) != '\n' && c != EOF);

    if (scene_choice == 2) {
        printf("Loading pre-defined natural scene...\n");

        // Ask for number of trees
        int num_trees;
        printf("Enter the number of trees to place in the scene (1 to 15): ");
        scanf("%d", &num_trees);
        while ((c = getchar()) != '\n' && c != EOF);
        
        if (num_trees < 1) num_trees = 1;
        if (num_trees > 15) {
            printf("Warning: Too many trees requested. Limiting to 15 trees to stay within object limits.\n");
            num_trees = 15;
        }

        // Ground - a large sphere
        Material_Device ground_mat = material_lambertian_create_host(vec3_create(0.5f, 0.5f, 0.2f));
        add_sphere_to_scene_host(vec3_create(0.0f, -1000.0f, -1.0f), 1000.0f, ground_mat);

        // Create materials for trees
        Material_Device trunk_mat = material_lambertian_create_host(vec3_create(0.4f, 0.2f, 0.1f));
        Material_Device canopy_mat = material_lambertian_create_host(vec3_create(0.1f, 0.5f, 0.1f));

        // Light source (sun) - Add this FIRST to guarantee it's always present
        Material_Device light_mat = material_emissive_create_host(vec3_scale(vec3_create(1.0f, 1.0f, 1.0f), 3.0f));
        Vec3 light_pos = vec3_create(3.0f, 3.0f, -1.0f);
        add_sphere_to_scene_host(light_pos, 0.5f, light_mat);
        printf("    -> Created Sun/Light source at position (%.2f, %.2f, %.2f)\n", light_pos.x, light_pos.y, light_pos.z);

        // Calculate max objects we can use for trees (reserve 1 for ground, 1 for light already added)
        int max_objects_for_trees = MAX_OBJECTS - 2; // Ground + Light already accounted for
        int estimated_objects_per_tree = 3; // 1 trunk + 2 canopy spheres minimum
        int safe_num_trees = (max_objects_for_trees) / estimated_objects_per_tree;
        
        if (num_trees > safe_num_trees) {
            printf("Warning: Adjusting number of trees from %d to %d to ensure all trees have complete canopies.\n", 
                   num_trees, safe_num_trees);
            num_trees = safe_num_trees;
        }

        // Generate trees at random positions
        Vec3 first_tree_pos = vec3_create(0, 0, 0); // Will store the first tree position for camera pivot
        for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
            // Generate random position for tree (spread them out on the ground)
            float x_pos = host_random_float_range(-8.0f, 8.0f);
            float z_pos = host_random_float_range(-8.0f, -2.0f); // Keep trees in front of camera
            Vec3 tree_base_pos = vec3_create(x_pos, 0.25f, z_pos);
            
            if (tree_idx == 0) {
                first_tree_pos = tree_base_pos; // Remember first tree for pivot
            }

            // Tree Trunk - a cube
            add_cube_to_scene_host(tree_base_pos, vec3_create(0.2f, 1.5f, 0.2f), trunk_mat);

            // Tree Canopy - multiple spheres for each tree
            Vec3 canopy_center = vec3_create(tree_base_pos.x, tree_base_pos.y + 0.95f, tree_base_pos.z);
            
            // Main canopy sphere - ALWAYS add this for every tree
            add_sphere_to_scene_host(canopy_center, 0.8f, canopy_mat);
            
            // Additional canopy spheres for fuller appearance (only if we have room)
            if (h_num_spheres + h_num_cubes < MAX_OBJECTS - 1) {
                Vec3 canopy_offset1 = vec3_create(canopy_center.x + host_random_float_range(-0.4f, 0.4f), 
                                                 canopy_center.y + host_random_float_range(-0.2f, 0.2f), 
                                                 canopy_center.z + host_random_float_range(-0.3f, 0.3f));
                add_sphere_to_scene_host(canopy_offset1, host_random_float_range(0.5f, 0.7f), canopy_mat);
            }
            
            if (h_num_spheres + h_num_cubes < MAX_OBJECTS) {
                Vec3 canopy_offset2 = vec3_create(canopy_center.x + host_random_float_range(-0.3f, 0.3f), 
                                                 canopy_center.y + host_random_float_range(-0.1f, 0.3f), 
                                                 canopy_center.z + host_random_float_range(-0.4f, 0.4f));
                add_sphere_to_scene_host(canopy_offset2, host_random_float_range(0.6f, 0.8f), canopy_mat);
            }

            printf("    -> Created Tree %d at position (%.2f, %.2f, %.2f)\n", 
                   tree_idx + 1, tree_base_pos.x, tree_base_pos.y, tree_base_pos.z);
        }

        // Set camera pivot to the first tree
        g_pivot_point_host = first_tree_pos;
        pivot_set_by_light = true; // Using this to signal pivot is intentionally set
        printf("    -> Camera will pivot around the first tree for rotation.\n");

    } else {
        int num_total_objects;
        printf("Enter the total number of objects for the scene (0 to %d): ", MAX_OBJECTS);
        scanf("%d", &num_total_objects);
        while ((c = getchar()) != '\n' && c != EOF);

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
            random_color_val.x = fmaxf(0.1f, random_color_val.x);
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

            if (shape_type_choice == 1) {
                float radius = host_random_float_range(0.3f, 0.7f);
                add_sphere_to_scene_host(center, radius, mat);
                printf("    -> Created Sphere at (%.2f, %.2f, %.2f), radius %.2f\n", center.x, center.y, center.z, radius);
            } else if (shape_type_choice == 2) {
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
    gpuErrchk(cudaMalloc((void**)&d_pixel_data, g_current_width * g_current_height * sizeof(Vec3)));
    gpuErrchk(cudaMalloc((void**)&d_rand_states, g_current_width * g_current_height * sizeof(curandState)));

    int num_rand_states = g_current_width * g_current_height;
    dim3 threadsPerBlockRand(256);
    dim3 numBlocksRand((num_rand_states + threadsPerBlockRand.x - 1) / threadsPerBlockRand.x);
    init_random_states_kernel<<<numBlocksRand, threadsPerBlockRand>>>(d_rand_states, num_rand_states, (unsigned long long)time(NULL));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void render_frame_cuda(SDL_Renderer *renderer, SDL_Texture *texture, int width, int height) {
    // Update resolution if it changed
    if (width != g_current_width || height != g_current_height) {
        resize_gpu_buffers(width, height);
    }
    
    camera_init_host(&d_camera, g_camera_pos_host, g_camera_lookat_host, g_camera_vup_host, g_fov_y_degrees_host, (float)g_current_width / g_current_height);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((g_current_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (g_current_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    render_kernel<<<numBlocks, threadsPerBlock>>>(d_pixel_data, g_current_width, g_current_height, d_camera,
                                                 d_spheres_data, h_num_spheres,
                                                 d_cubes_data, h_num_cubes,
                                                 d_rand_states);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    Vec3* h_pixels = (Vec3*)malloc(g_current_width * g_current_height * sizeof(Vec3));
    gpuErrchk(cudaMemcpy(h_pixels, d_pixel_data, g_current_width * g_current_height * sizeof(Vec3), cudaMemcpyDeviceToHost));

    void *sdl_pixels_locked;
    int pitch;
    SDL_LockTexture(texture, NULL, &sdl_pixels_locked, &pitch);
    unsigned char *pixel_data_sdl = (unsigned char *)sdl_pixels_locked;

    for (int y = 0; y < g_current_height; y++) {
        for (int x = 0; x < g_current_width; x++) {
            Vec3 p_color = h_pixels[y * g_current_width + x];
            int ir = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.x)));
            int ig = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.y)));
            int ib = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.z)));

            int index = y * pitch + x * 4;
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

void resize_gpu_buffers(int new_width, int new_height) {
    printf("Resizing GPU buffers from %dx%d to %dx%d\n", g_current_width, g_current_height, new_width, new_height);
    
    // Free old buffers
    if (d_pixel_data) gpuErrchk(cudaFree(d_pixel_data));
    if (d_rand_states) gpuErrchk(cudaFree(d_rand_states));
    
    // Update resolution
    g_current_width = new_width;
    g_current_height = new_height;
    
    // Allocate new buffers
    gpuErrchk(cudaMalloc((void**)&d_pixel_data, g_current_width * g_current_height * sizeof(Vec3)));
    gpuErrchk(cudaMalloc((void**)&d_rand_states, g_current_width * g_current_height * sizeof(curandState)));

    // Reinitialize random states
    int num_rand_states = g_current_width * g_current_height;
    dim3 threadsPerBlockRand(256);
    dim3 numBlocksRand((num_rand_states + threadsPerBlockRand.x - 1) / threadsPerBlockRand.x);
    init_random_states_kernel<<<numBlocksRand, threadsPerBlockRand>>>(d_rand_states, num_rand_states, (unsigned long long)time(NULL));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
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