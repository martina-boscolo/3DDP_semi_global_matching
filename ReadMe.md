SGM-Census with mono-depth initial guess
========================================

---
## Instructions

**Building (Out-of-Tree)**

    mkdir build
    cd build/
    cmake ..
    make
    
**Usage (from build/ directory)** 
Notice the requirement for also the monocular_left image!

    ./sgm <right image> <left image> <monocular_right> <monocular_left> <gt disparity map> <output image file> <disparity range> 

**Examples**

    ./sgm ../Examples/Rocks1/right.png ../Examples/Rocks1/left.png ../Examples/Rocks1/right_mono.png ../Examples/Rocks1/left_mono.png ../Examples/Rocks1/rightGT.png ../Examples/Rocks1/output_disparity.png 85

    ./sgm ../Examples/Plastic/right.png ../Examples/Plastic/left.png ../Examples/Plastic/right_mono.png ../Examples/Plastic/left_mono.png ../Examples/Plastic/rightGT.png ../Examples/Plastic/output_disparity.png 85

    ./sgm ../Examples/Cones/right.png ../Examples/Cones/left.png ../Examples/Cones/right_mono.png ../Examples/Cones/left_mono.png ../Examples/Cones/rightGT.png ../Examples/Cones/output_disparity.png 85

    ./sgm ../Examples/Aloe/right.png ../Examples/Aloe/left.png ../Examples/Aloe/right_mono.png ../Examples/Aloe/left_mono.png ../Examples/Aloe/rightGT.png ../Examples/Aloe/output_disparity.png 85
---


