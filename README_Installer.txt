RealTimeParticles 1.0

Minimalist real-time 3D particles simulator combining DearImGui/OpenGL and OpenCL to admire and play with small-sized (512) to large (130k) particle systems. 

# Real-time physics models implemented

1. Boids based on classical Craig Reynolds implementation (https://www.red3d.com/cwr/boids/).
2. Position Based Fluids based on NVIDIA paper, Macklin and Muller, 2013. "Position Based Fluids" (https://mmacklin.com/pbf_sig_preprint.pdf).

# Requirements

- Device (GPU, IGPU or else) supporting OpenGL and OpenCL 1.2 or higher.
Be advised, for best performance make sure that the application runs on a discrete GPU and not default IGPU.

# References

- [CMake](https://cmake.org/)
- [ImGui](https://github.com/ocornut/imgui)
- [Conan](https://conan.io/)
- [OpenCL](https://www.khronos.org/opencl/)
- [SDL2](https://libsdl.org/index.php)
- [Glad](https://glad.dav1d.de/)
- [spdlog](https://github.com/gabime/spdlog)
- [NSIS](http://nsis.sourceforge.net/)
- [OpenCL radix sort](https://github.com/modelflat/OCLRadixSort)
- [Simon Green N-body simulation paper](https://developer.download.nvidia.com/assets/cuda/files/particles.pdf)
- [Perlin Noise C++ implementation](https://github.com/sol-prog/Perlin_Noise)

# Notes

Whole application has been tested only on a handful of Windows machines. I will happily make sure it works on Linux machines once I have access to one.
Concerning performance, I reach 60fps with a Nvidia GTX 1650 for the 130k boids 3D model, and 12-30fps for the 3D fluids simulation Dam depending on selected settings.

Please feel free to contact me if you want to provide some feedback or need troubleshooting : moulin.adrien00@gmail.com / https://github.com/axoloto