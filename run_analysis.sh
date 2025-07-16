#!/bin/bash

echo "================================================"
echo "Ray Tracing Performance Analysis Pipeline"
echo "================================================"

# Check if required tools are available
if ! command -v nvcc &> /dev/null; then
    echo "Error: NVCC (NVIDIA CUDA Compiler) not found!"
    echo "Please install CUDA toolkit first."
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
make -f Makefile.perf clean-perf

# Build the performance analysis tool
echo "Building performance analysis tool..."
if ! make -f Makefile.perf perf; then
    echo "Error: Failed to build performance analysis tool!"
    exit 1
fi

# Run the performance analysis
echo "Running performance analysis..."
echo "This may take a few minutes..."
if ! ./bin/performance_analysis; then
    echo "Error: Performance analysis failed!"
    exit 1
fi

echo ""
echo "================================================"
echo "Performance Analysis Complete!"
echo "================================================"
echo "Results:"
echo "- Raw data: performance_analysis.csv"
echo "- Report: ANALYSIS_REPORT.md"
echo ""
echo "To view the report:"
echo "  cat ANALYSIS_REPORT.md"
echo "  cat ANALYSIS_REPORT.md"
echo "  # or open ANALYSIS_REPORT.md in a markdown viewer"
echo ""
echo "Key findings:"
echo "- Check speedup values for performance improvements"
echo "- Check RMSE values for accuracy (lower is better)"
echo "- Check PSNR values for signal quality (higher is better)"
