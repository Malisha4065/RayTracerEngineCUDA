#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <SDL2/SDL.h>

#include "../include/scene.h"
#include "../include/cuda_utils.h"

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));

    gpuErrchk(cudaFree(0)); 
    size_t new_stack_size = 16384;
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
            float cam_offset_z = g_distance_to_pivot_host * cosf(g_camera_pitch_host) * -cosf(g_camera_yaw_host);
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
    }

    cleanup_gpu_data();
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}