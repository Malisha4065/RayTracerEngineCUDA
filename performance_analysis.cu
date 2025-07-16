#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Include the headers from your project
#include "include/scene.h"
#include "include/kernels.h"
#include "include/cuda_utils.h"

// Test configurations
#define TEST_WIDTH 800
#define TEST_HEIGHT 600
#define NUM_ITERATIONS 5

// Simple CPU ray tracer for comparison
struct SimpleRay {
    Vec3 origin, direction;
};

struct SimpleHit {
    Vec3 point, normal;
    float t;
    bool hit;
};

// Simple CPU sphere intersection
SimpleHit intersect_sphere_cpu(const SimpleRay& ray, Vec3 center, float radius) {
    SimpleHit hit;
    hit.hit = false;
    
    Vec3 oc = vec3_sub(ray.origin, center);
    float a = vec3_dot(ray.direction, ray.direction);
    float b = 2.0f * vec3_dot(oc, ray.direction);
    float c = vec3_dot(oc, oc) - radius * radius;
    
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return hit;
    
    float t = (-b - sqrtf(discriminant)) / (2.0f * a);
    if (t > 0.001f) {
        hit.hit = true;
        hit.t = t;
        hit.point = vec3_add(ray.origin, vec3_scale(ray.direction, t));
        hit.normal = vec3_normalize(vec3_sub(hit.point, center));
    }
    
    return hit;
}

// Simple CPU ray color calculation
Vec3 ray_color_cpu_simple(const SimpleRay& ray) {
    // Simple scene with one sphere
    SimpleHit hit = intersect_sphere_cpu(ray, vec3_create(0, 0, -1), 0.5f);
    
    if (hit.hit) {
        // Simple diffuse shading
        Vec3 light_dir = vec3_normalize(vec3_create(1, 1, 1));
        float dot_product = fmaxf(0.0f, vec3_dot(hit.normal, light_dir));
        return vec3_scale(vec3_create(0.7f, 0.3f, 0.3f), dot_product);
    }
    
    // Background gradient
    Vec3 unit_direction = vec3_normalize(ray.direction);
    float t = 0.5f * (unit_direction.y + 1.0f);
    Vec3 white = vec3_create(1.0f, 1.0f, 1.0f);
    Vec3 blue = vec3_create(0.5f, 0.7f, 1.0f);
    return vec3_add(vec3_scale(white, 1.0f - t), vec3_scale(blue, t));
}

// CPU renderer
void render_cpu_simple(Vec3* output, int width, int height) {
    // Simple camera setup
    Vec3 camera_pos = vec3_create(0, 0, 0);
    Vec3 lower_left = vec3_create(-2, -1, -1);
    Vec3 horizontal = vec3_create(4, 0, 0);
    Vec3 vertical = vec3_create(0, 2, 0);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float u = (float)x / (float)width;
            float v = (float)y / (float)height;
            
            SimpleRay ray;
            ray.origin = camera_pos;
            ray.direction = vec3_normalize(vec3_sub(vec3_add(vec3_add(lower_left, 
                                                                      vec3_scale(horizontal, u)), 
                                                             vec3_scale(vertical, v)), 
                                                    camera_pos));
            
            output[y * width + x] = ray_color_cpu_simple(ray);
        }
    }
}

// Calculate Mean Squared Error between two images
double calculate_mse(Vec3* img1, Vec3* img2, int width, int height) {
    double sum_squared_error = 0.0;
    int total_pixels = width * height;
    
    for (int i = 0; i < total_pixels; i++) {
        double dr = img1[i].x - img2[i].x;
        double dg = img1[i].y - img2[i].y;
        double db = img1[i].z - img2[i].z;
        
        sum_squared_error += dr * dr + dg * dg + db * db;
    }
    
    return sum_squared_error / (3.0 * total_pixels);
}

// Calculate RMSE
double calculate_rmse(Vec3* img1, Vec3* img2, int width, int height) {
    return sqrt(calculate_mse(img1, img2, width, height));
}

// Performance test function
void run_performance_test(int threads_x, int threads_y, FILE* csv_file) {
    printf("Testing with %dx%d thread blocks...\n", threads_x, threads_y);
    
    // Allocate host memory
    Vec3* cpu_result = (Vec3*)malloc(TEST_WIDTH * TEST_HEIGHT * sizeof(Vec3));
    Vec3* gpu_result = (Vec3*)malloc(TEST_WIDTH * TEST_HEIGHT * sizeof(Vec3));
    
    // Allocate device memory
    Vec3* d_framebuffer;
    gpuErrchk(cudaMalloc(&d_framebuffer, TEST_WIDTH * TEST_HEIGHT * sizeof(Vec3)));
    
    // Initialize random states
    curandState* d_rand_states;
    int num_states = TEST_WIDTH * TEST_HEIGHT;
    gpuErrchk(cudaMalloc(&d_rand_states, num_states * sizeof(curandState)));
    
    dim3 rand_block_size(256);
    dim3 rand_grid_size((num_states + rand_block_size.x - 1) / rand_block_size.x);
    
    init_random_states_kernel<<<rand_grid_size, rand_block_size>>>(d_rand_states, num_states, time(NULL));
    gpuErrchk(cudaDeviceSynchronize());
    
    // Setup camera
    Camera_Device cam;
    cam.origin = vec3_create(0, 0, 0);
    cam.lower_left_corner = vec3_create(-2, -1, -1);
    cam.horizontal = vec3_create(4, 0, 0);
    cam.vertical = vec3_create(0, 2, 0);
    
    // CPU timing
    double cpu_total_time = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        render_cpu_simple(cpu_result, TEST_WIDTH, TEST_HEIGHT);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        cpu_total_time += cpu_time;
    }
    double avg_cpu_time = cpu_total_time / NUM_ITERATIONS;
    
    // GPU timing
    dim3 block_size(threads_x, threads_y);
    dim3 grid_size((TEST_WIDTH + block_size.x - 1) / block_size.x, 
                   (TEST_HEIGHT + block_size.y - 1) / block_size.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float gpu_total_time = 0.0f;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaEventRecord(start);
        
        render_kernel<<<grid_size, block_size>>>(d_framebuffer, TEST_WIDTH, TEST_HEIGHT, cam,
                                                d_spheres_data, h_num_spheres,
                                                d_cubes_data, h_num_cubes,
                                                d_rand_states);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start, stop);
        gpu_total_time += gpu_time;
    }
    float avg_gpu_time = gpu_total_time / NUM_ITERATIONS;
    
    // Copy final result back to host
    gpuErrchk(cudaMemcpy(gpu_result, d_framebuffer, TEST_WIDTH * TEST_HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost));
    
    // Calculate metrics
    double mse = calculate_mse(cpu_result, gpu_result, TEST_WIDTH, TEST_HEIGHT);
    double rmse = sqrt(mse);
    double psnr = 10.0 * log10(3.0 / mse); // Peak Signal-to-Noise Ratio
    double speedup = avg_cpu_time / avg_gpu_time;
    
    // Print results
    printf("CPU Time: %.2f ms\n", avg_cpu_time);
    printf("GPU Time: %.2f ms\n", avg_gpu_time);
    printf("Speedup: %.2fx\n", speedup);
    printf("MSE: %.6f\n", mse);
    printf("RMSE: %.6f\n", rmse);
    printf("PSNR: %.2f dB\n", psnr);
    printf("----------------------\n");
    
    // Write to CSV
    fprintf(csv_file, "%dx%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f\n",
            threads_x, threads_y, threads_x, threads_y, threads_x * threads_y,
            grid_size.x * grid_size.y,
            avg_cpu_time, avg_gpu_time, speedup, mse, rmse, psnr);
    
    // Cleanup
    free(cpu_result);
    free(gpu_result);
    cudaFree(d_framebuffer);
    cudaFree(d_rand_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("Ray Tracing Performance Analysis\n");
    printf("================================\n");
    printf("Image Resolution: %dx%d\n", TEST_WIDTH, TEST_HEIGHT);
    printf("Number of iterations per test: %d\n", NUM_ITERATIONS);
    
    // Initialize CUDA and scene
    gpuErrchk(cudaFree(0));
    size_t stack_size = 16384;
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
    
    init_engine_scene_and_gpu_data();
    
    // Open CSV file for results
    FILE* csv_file = fopen("performance_analysis.csv", "w");
    fprintf(csv_file, "ThreadConfig,Threads_X,Threads_Y,ThreadsPerBlock,NumBlocks,CPU_Time_ms,GPU_Time_ms,Speedup,MSE,RMSE,PSNR_dB\n");
    
    // Test different thread configurations
    std::vector<std::pair<int, int>> thread_configs = {
        {1, 1},     // Serial-like execution
        {8, 8},     // 64 threads per block
        {16, 16},   // 256 threads per block
        {32, 32},   // 1024 threads per block
        {16, 32},   // 512 threads per block
        {32, 16},   // 512 threads per block (different aspect ratio)
        {64, 16},   // 1024 threads per block (wide)
        {16, 64},   // 1024 threads per block (tall)
    };
    
    for (const auto& config : thread_configs) {
        if (config.first * config.second <= 1024) { // Respect max threads per block
            run_performance_test(config.first, config.second, csv_file);
        }
    }
    
    fclose(csv_file);
    
    // Generate summary report
    printf("\n=== ANALYSIS SUMMARY ===\n");
    printf("Performance analysis complete!\n");
    printf("Results saved to: performance_analysis.csv\n");
    printf("\nKey Metrics Explained:\n");
    printf("- Speedup: How much faster GPU is compared to CPU\n");
    printf("- MSE: Mean Squared Error (lower is better)\n");
    printf("- RMSE: Root Mean Squared Error (lower is better)\n");
    printf("- PSNR: Peak Signal-to-Noise Ratio (higher is better)\n");
    printf("\nOptimal configuration should show:\n");
    printf("- High speedup (>10x)\n");
    printf("- Low RMSE (<0.1)\n");
    printf("- High PSNR (>20 dB)\n");
    
    cleanup_gpu_data();
    
    return 0;
}
