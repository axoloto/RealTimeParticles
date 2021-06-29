# Particle System Sandbox

Minimalist particle simulator combining DearImGui/OpenGL and OpenCL to admire and play with small-sized (256) to large (260k) particle systems. 

Physics models :
- Boids, currently based on classical Reynolds implementation
- SPH fluids (to come)

**For best performance, make sure that the application runs on any available discrete GPU and not the IGPU.**

![](boids.gif)

# Requirements

- Gitbash (https://git-scm.com/downloads)
- Python >= 3.5 (https://www.python.org/) and pip (https://pypi.org/project/pip/)
- Conan (https://conan.io/)
- CMake (https://cmake.org/download/)
- C++ compiler, tested with MSVC 15/19 only for now
- Device (GPU or IGPU) supporting OpenGL and OpenCL 1.2 or higher

# Build and Run

```bash
pip install conan
conan remote add bincrafters "https://api.bintray.com/conan/bincrafters/public-conan"
git clone https://github.com/axoloto/Boids.git
cd Boids
./runApp.sh
```

# References

- Boids by Craig Reynolds (https://www.red3d.com/cwr/boids/)

- CMake (https://cmake.org/)
- ImGui (https://github.com/ocornut/imgui)
- OpenCL (https://www.khronos.org/opencl/)
- SDL2 (https://libsdl.org/index.php)
- Glad (https://glad.dav1d.de/)
- spdlog (https://github.com/gabime/spdlog)

# Notes

Whole application has been tested only on a handful of Windows machines. I will happily make sure it works on Linux machines as well once I have access to one. Concerning performance, with 260k particles in 3D and default boids settings, I reach 60fps with a Nvidia GTX 1650.
