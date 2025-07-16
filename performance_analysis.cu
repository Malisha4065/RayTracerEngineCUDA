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
#include "include/common_math.h" // Add for MAX_DEPTH

// Test configurations
#define TEST_WIDTH 800
#define TEST_HEIGHT 600
#define NUM_ITERATIONS 5
#define SAMPLES_PER_PIXEL 4

// Simple CPU ray tracer for comparison
struct SimpleRay {
    Vec3 origin, direction;
};

struct SimpleHit {
    Vec3 point, normal;
    float t;
    Material_Device material; // Store material of hit object
    bool hit;
};

// Simple random number generator to match GPU behavior
class SimpleRandom {
private:
    unsigned long long state;
public:
    SimpleRandom(unsigned long long s) : state(s) {}
    
    // LCG implementation similar to curand_uniform
    float random_float() {
        state = state * 1103515245ULL + 12345ULL;
        return ((state >> 16) & 0xFFFFFFFF) / 4294967296.0f;
    }
    
    float random_float_range(float min, float max) {
        return min + (max - min) * random_float();
    }
    
    Vec3 random_in_unit_sphere() {
        while (true) {
            Vec3 p = vec3_create(random_float_range(-1.0f, 1.0f),
                                random_float_range(-1.0f, 1.0f),
                                random_float_range(-1.0f, 1.0f));
            if (vec3_length_squared(p) < 1.0f) return p;
        }
    }
    
    Vec3 random_unit_vector() {
        return vec3_normalize(random_in_unit_sphere());
    }
};

// Helper to generate a random point in a unit sphere for diffuse bounces
Vec3 random_in_unit_sphere_cpu() {
    Vec3 p;
    do {
        p = vec3_create(2.0f * (float)rand() / RAND_MAX - 1.0f,
                        2.0f * (float)rand() / RAND_MAX - 1.0f,
                        2.0f * (float)rand() / RAND_MAX - 1.0f);
    } while (vec3_dot(p, p) >= 1.0f);
    return p;
}

// Simple CPU sphere intersection
SimpleHit intersect_sphere_cpu(const SimpleRay& ray, const SphereData_Device& sphere) {
    SimpleHit hit;
    hit.hit = false;
    
    Vec3 oc = vec3_sub(ray.origin, sphere.center);
    float a = vec3_dot(ray.direction, ray.direction);
    float b = 2.0f * vec3_dot(oc, ray.direction);
    float c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return hit;
    
    float t = (-b - sqrtf(discriminant)) / (2.0f * a);
    if (t > 0.001f) {
        hit.hit = true;
        hit.t = t;
        hit.point = vec3_add(ray.origin, vec3_scale(ray.direction, t));
        hit.normal = vec3_normalize(vec3_sub(hit.point, sphere.center));
        hit.material = sphere.material; // Copy material info
    }
    
    return hit;
}

// CPU path tracing function to match the GPU kernel's logic
Vec3 ray_color_cpu_path_trace(SimpleRay ray, const SphereData_Device* spheres, int num_spheres, SimpleRandom* rng) {
    Vec3 attenuation = vec3_create(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        SimpleHit closest_hit;
        closest_hit.hit = false;
        closest_hit.t = INFINITY;

        // Find the closest intersection among all spheres
        for (int i = 0; i < num_spheres; i++) {
            SimpleHit hit = intersect_sphere_cpu(ray, spheres[i]);
            if (hit.hit && hit.t < closest_hit.t) {
                closest_hit = hit;
            }
        }

        if (closest_hit.hit) {
            // Lambertian scatter (matches the GPU logic)
            Vec3 scatter_direction = vec3_add(closest_hit.normal, rng->random_unit_vector());
            if (vec3_length_squared(scatter_direction) < 1e-8f) {
                scatter_direction = closest_hit.normal;
            }
            
            // Update the ray for the next bounce
            ray.origin = closest_hit.point;
            ray.direction = vec3_normalize(scatter_direction);
            
            // Update the color attenuation
            attenuation = vec3_mul(attenuation, closest_hit.material.albedo);
        } else {
            // Ray missed all objects, return background color multiplied by current attenuation
            Vec3 unit_direction = vec3_normalize(ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            Vec3 white = vec3_create(1.0f, 1.0f, 1.0f);
            Vec3 blue = vec3_create(0.5f, 0.7f, 1.0f);
            Vec3 background_color = vec3_add(vec3_scale(white, 1.0f - t), vec3_scale(blue, t));
            return vec3_mul(attenuation, background_color);
        }
    }

    // Max depth reached, return black
    return vec3_create(0.0f, 0.0f, 0.0f);
}

// CPU renderer - renders using the same camera and sampling logic as GPU
void render_cpu_simple(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = y * width + x;
            // Initialize random state per pixel to match GPU behavior
            SimpleRandom rng(base_seed + pixel_index);
            
            Vec3 pixel_color = vec3_create(0.0f, 0.0f, 0.0f);
            
            // Multi-sample anti-aliasing to match GPU
            for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                // Calculate UV coordinates with same logic as GPU
                float u = (float)(x + ((SAMPLES_PER_PIXEL > 1) ? rng.random_float() : 0.5f)) / (width - 1);
                float v_img = (float)(y + ((SAMPLES_PER_PIXEL > 1) ? rng.random_float() : 0.5f)) / (height - 1);
                float v_cam = 1.0f - v_img; // Flip Y coordinate to match GPU
                
                // Generate ray using same camera logic as GPU
                SimpleRay ray;
                ray.origin = camera.origin;
                ray.direction = vec3_normalize(vec3_sub(vec3_add(vec3_add(camera.lower_left_corner, 
                                                                          vec3_scale(camera.horizontal, u)), 
                                                                 vec3_scale(camera.vertical, v_cam)), 
                                                        camera.origin));
                
                // Trace ray and accumulate color
                pixel_color = vec3_add(pixel_color, ray_color_cpu_path_trace(ray, spheres, num_spheres, &rng));
            }
            
            // Average the samples
            pixel_color = vec3_div(pixel_color, (float)SAMPLES_PER_PIXEL);
            
            // Apply gamma correction to match GPU
            pixel_color.x = sqrtf(fmaxf(0.0f, pixel_color.x));
            pixel_color.y = sqrtf(fmaxf(0.0f, pixel_color.y));
            pixel_color.z = sqrtf(fmaxf(0.0f, pixel_color.z));
            
            output[pixel_index] = pixel_color;
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
    
    // Calculate grid dimensions
    dim3 block_size(threads_x, threads_y);
    dim3 grid_size((TEST_WIDTH + block_size.x - 1) / block_size.x, 
                   (TEST_HEIGHT + block_size.y - 1) / block_size.y);
    
    int threads_per_block = threads_x * threads_y;
    int total_blocks = grid_size.x * grid_size.y;
    int total_threads = threads_per_block * total_blocks;
    
    printf("  Block size: %dx%d (%d threads per block)\n", threads_x, threads_y, threads_per_block);
    printf("  Grid size: %dx%d (%d blocks total)\n", grid_size.x, grid_size.y, total_blocks);
    printf("  Total threads: %d\n", total_threads);
    
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
    
    unsigned int base_seed = 1234; // Fixed seed for reproducibility
    init_random_states_kernel<<<rand_grid_size, rand_block_size>>>(d_rand_states, num_states, base_seed);
    gpuErrchk(cudaDeviceSynchronize());
    
    // Setup camera - use same configuration for both CPU and GPU
    Camera_Device cam;
    cam.origin = vec3_create(0, 0, 0);
    cam.lower_left_corner = vec3_create(-2, -1.5, -1);
    cam.horizontal = vec3_create(4, 0, 0);
    cam.vertical = vec3_create(0, 3, 0);
    
    // Create simple test scene - same for both CPU and GPU
    SphereData_Device* d_simple_spheres;
    gpuErrchk(cudaMalloc(&d_simple_spheres, 3 * sizeof(SphereData_Device)));
    
    SphereData_Device simple_spheres[3];
    // Ground sphere
    simple_spheres[0].center = vec3_create(0, -100.5f, -1);
    simple_spheres[0].radius = 100.0f;
    simple_spheres[0].material = material_lambertian_create_host(vec3_create(0.5f, 0.5f, 0.5f));
    
    // Center sphere
    simple_spheres[1].center = vec3_create(0, 0, -1);
    simple_spheres[1].radius = 0.5f;
    simple_spheres[1].material = material_lambertian_create_host(vec3_create(0.7f, 0.3f, 0.3f));
    
    // Left sphere
    simple_spheres[2].center = vec3_create(-1, 0, -1);
    simple_spheres[2].radius = 0.5f;
    simple_spheres[2].material = material_lambertian_create_host(vec3_create(0.8f, 0.8f, 0.0f));
    
    gpuErrchk(cudaMemcpy(d_simple_spheres, simple_spheres, 3 * sizeof(SphereData_Device), cudaMemcpyHostToDevice));

    // CPU timing
    double cpu_total_time = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        render_cpu_simple(cpu_result, TEST_WIDTH, TEST_HEIGHT, simple_spheres, 3, cam, base_seed);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        cpu_total_time += cpu_time;
    }
    double avg_cpu_time = cpu_total_time / NUM_ITERATIONS;
    
    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float gpu_total_time = 0.0f;
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cudaEventRecord(start);
        
        render_kernel<<<grid_size, block_size>>>(d_framebuffer, TEST_WIDTH, TEST_HEIGHT, cam,
                                                d_simple_spheres, 3,  // Use 3 spheres
                                                nullptr, 0,           // No cubes
                                                d_rand_states);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Check for kernel errors
        cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess) {
            printf("CUDA kernel error: %s\n", cudaGetErrorString(kernel_error));
        }
        
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start, stop);
        gpu_total_time += gpu_time;
    }
    
    // Clean up simple scene data
    cudaFree(d_simple_spheres);
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
    
    // Warn about suspicious results
    if (avg_gpu_time < 1.0f) {
        printf("WARNING: GPU time suspiciously low - kernel may not be executing properly!\n");
    }
    if (mse > 0.1) {
        printf("WARNING: High MSE detected - results may be inaccurate!\n");
    }
    
    printf("----------------------\n");
    
    // Write to CSV
    fprintf(csv_file, "%dx%d,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f\n",
            threads_x, threads_y, threads_x, threads_y, threads_per_block,
            grid_size.x, grid_size.y, total_blocks, total_threads,
            avg_cpu_time, avg_gpu_time, speedup, mse, rmse, psnr);
    
    // Cleanup
    free(cpu_result);
    free(gpu_result);
    cudaFree(d_framebuffer);
    cudaFree(d_rand_states);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}int main() {
    printf("Ray Tracing Performance Analysis\n");
    printf("================================\n");
    printf("Image Resolution: %dx%d\n", TEST_WIDTH, TEST_HEIGHT);
    printf("Number of iterations per test: %d\n", NUM_ITERATIONS);
    
    // Query device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    printf("\nGPU Device Information:\n");
    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max block dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Total max threads: %d\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("================================\n\n");
    
    // Initialize CUDA and scene
    gpuErrchk(cudaFree(0));
    size_t stack_size = 16384;
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
    
    // We don't need to initialize the complex scene for this test
    // The test function will create its own simple scene
    
    // Open CSV file for results
    FILE* csv_file = fopen("performance_analysis.csv", "w");
    fprintf(csv_file, "ThreadConfig,Threads_X,Threads_Y,ThreadsPerBlock,Grid_X,Grid_Y,NumBlocks,TotalThreads,CPU_Time_ms,GPU_Time_ms,Speedup,MSE,RMSE,PSNR_dB\n");
    
    // Test different thread configurations optimized for MX330
    // Max threads per block: 1024, but optimal performance usually at 256-512
    // Warp size: 32, so use multiples of 32 for best efficiency
    std::vector<std::pair<int, int>> thread_configs = {
        {1, 1},     // Serial-like execution (1 thread per block) - many blocks
        {2, 2},     // 4 threads per block
        {4, 4},     // 16 threads per block
        {8, 8},     // 64 threads per block (2 warps)
        {16, 16},   // 256 threads per block (8 warps) - good for MX330
        {16, 32},   // 512 threads per block (16 warps) - good for MX330
        {32, 16},   // 512 threads per block (16 warps, different aspect ratio)
        {8, 16},    // 128 threads per block (4 warps)
        {16, 8},    // 128 threads per block (4 warps, different aspect ratio)
        {4, 16},    // 64 threads per block (2 warps, tall)
        {16, 4},    // 64 threads per block (2 warps, wide)
        {32, 8},    // 256 threads per block (8 warps, wide)
        {8, 32},    // 256 threads per block (8 warps, tall)
        {32, 4},    // 128 threads per block (4 warps, very wide)
        {4, 32},    // 128 threads per block (4 warps, very tall)
        // Avoiding 32x32 and other high thread counts as they may cause issues on MX330
    };
    
    for (const auto& config : thread_configs) {
        int threads_per_block = config.first * config.second;
        
        // Check device constraints
        bool valid_config = true;
        valid_config &= (threads_per_block <= prop.maxThreadsPerBlock);
        valid_config &= (config.first <= prop.maxThreadsDim[0]);
        valid_config &= (config.second <= prop.maxThreadsDim[1]);
        
        // For MX330, avoid very high thread counts per block as they may not perform well
        if (prop.multiProcessorCount <= 4) { // Low-end GPU
            valid_config &= (threads_per_block <= 512); // Conservative limit
        }
        
        if (valid_config) {
            run_performance_test(config.first, config.second, csv_file);
        } else {
            printf("Skipping %dx%d configuration (not suitable for device: %d threads per block)\n", 
                   config.first, config.second, threads_per_block);
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
    printf("\nGPU Thread Organization:\n");
    printf("- Each thread processes exactly 1 pixel\n");
    printf("- Total threads = (Grid_X × Grid_Y) × (Threads_X × Threads_Y)\n");
    printf("- For %dx%d image, we need %d total threads\n", TEST_WIDTH, TEST_HEIGHT, TEST_WIDTH * TEST_HEIGHT);
    printf("- MX330 has %d SMs with max %d threads each = %d total threads\n", 
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, 
           prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("- Optimal block size is usually 256-512 threads for MX330\n");
    printf("- Use multiples of %d (warp size) for best efficiency\n", prop.warpSize);
    printf("\nOptimal configuration should show:\n");
    printf("- High speedup (>10x for MX330)\n");
    printf("- Low RMSE (<0.1)\n");
    printf("- High PSNR (>20 dB)\n");
    printf("- GPU time > 1ms (avoid suspiciously low times)\n");
    printf("- Efficient use of GPU resources\n");
    
    return 0;
}
