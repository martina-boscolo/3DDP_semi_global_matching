#!/bin/bash
# filepath: /home/tdp/Desktop/3D_data_processing/run_all.sh

# Navigate to build directory, assuming script is run from project root
cd build || { echo "Error: build directory not found. Please run from project root after building."; exit 1; }

echo "Running SGM on all example datasets..."

# Rocks1 dataset
echo "Processing Rocks1 dataset..."
./sgm ../Examples/Rocks1/right.png ../Examples/Rocks1/left.png ../Examples/Rocks1/right_mono.png ../Examples/Rocks1/left_mono.png ../Examples/Rocks1/rightGT.png ../Examples/Rocks1/output_disparity.png 85

# Plastic dataset
echo "Processing Plastic dataset..."
./sgm ../Examples/Plastic/right.png ../Examples/Plastic/left.png ../Examples/Plastic/right_mono.png ../Examples/Plastic/left_mono.png ../Examples/Plastic/rightGT.png ../Examples/Plastic/output_disparity.png 85

# Cones dataset
echo "Processing Cones dataset..."
./sgm ../Examples/Cones/right.png ../Examples/Cones/left.png ../Examples/Cones/right_mono.png ../Examples/Cones/left_mono.png ../Examples/Cones/rightGT.png ../Examples/Cones/output_disparity.png 85

# Aloe dataset
echo "Processing Aloe dataset..."
./sgm ../Examples/Aloe/right.png ../Examples/Aloe/left.png ../Examples/Aloe/right_mono.png ../Examples/Aloe/left_mono.png ../Examples/Aloe/rightGT.png ../Examples/Aloe/output_disparity.png 85

echo "All examples processed successfully."