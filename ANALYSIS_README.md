# Ray Tracing Performance Analysis

This analysis provides performance and accuracy results for the CUDA ray tracer implementation.

## Quick Start

To run the complete analysis:

```bash
./run_analysis.sh
```

## Output Files

- `performance_analysis.csv` - Raw performance data

## Analysis Components

### 1. Performance Measurement

The analysis measures:
- **CPU vs GPU timing** across different thread configurations
- **Speedup calculations** (how many times faster GPU is than CPU)
- **Thread block optimization** (1x1 to 32x32 configurations)
- **Memory access patterns** impact on performance

### 2. Accuracy Validation

The analysis validates:
- **RMSE (Root Mean Square Error)** between CPU and GPU results
- **PSNR (Peak Signal-to-Noise Ratio)** for image quality assessment
- **Pixel-level accuracy** comparison
- **Numerical precision** of parallel implementation

### 3. Thread Configuration Analysis

Tests various thread block sizes:
- 1x1 (serial-like)
- 8x8 (64 threads per block)
- 16x16 (256 threads per block)
- 32x32 (1024 threads per block)
- 16x32, 32x16 (different aspect ratios)

## Key Metrics

### Performance Metrics
- **Speedup**: GPU time / CPU time (higher is better)
- **GPU Execution Time**: Time in milliseconds for GPU rendering
- **Threads per Block**: Number of threads in each CUDA block
- **Number of Blocks**: Total CUDA blocks launched

### Accuracy Metrics
- **MSE (Mean Squared Error)**: Average squared difference between CPU and GPU pixels
- **RMSE (Root Mean Square Error)**: Square root of MSE (lower is better)
- **PSNR (Peak Signal-to-Noise Ratio)**: Signal quality metric in dB (higher is better)

## Expected Results

### Performance
- **Speedup**: 10-50x faster than CPU (depending on GPU)
- **Optimal thread configuration**: Usually 16x16 or 32x32 blocks
- **Diminishing returns**: Beyond optimal thread count

### Accuracy
- **RMSE**: Should be < 0.1 for good accuracy
- **PSNR**: Should be > 20 dB for good image quality
- **Consistency**: All configurations should show similar accuracy

## Understanding the Results

### Good Results
- Speedup > 10x: Excellent parallelization
- RMSE < 0.1: High accuracy
- PSNR > 20 dB: Good image quality

### Poor Results
- Speedup < 5x: Check thread configuration
- RMSE > 0.5: Potential accuracy issues
- PSNR < 15 dB: Image quality problems
