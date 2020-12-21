# Particle System Sandbox

Minimalist simulator combining ImGui/OpenGL and OpenCL to admire and play with medium-sized particle systems (30k). 

Physics models :
- Boids, currently based on classical Reynolds implementation
- SPH fluids (to come)

![](boids.gif)

# Requirements

- Windows/Linux
- Conan (https://conan.io/)
- Python >= 3.5 (https://www.python.org/) and pip (https://pypi.org/project/pip/)
- Device (GPU or IGPU) supporting OpenGL and OpenCL 1.2 or higher

# Build

```bash
$ pip install conan && \
$ conan remote add bincrafters "https://api.bintray.com/conan/bincrafters/public-conan"
$ git clone https://github.com/axoloto/Boids.git
$ cd boids 
$ ./runBuild.sh
$ ./runApp.sh
```

# References

- Boids by Craig Reynolds (https://www.red3d.com/cwr/boids/)

- CMake (https://cmake.org/)
- ImGui (https://github.com/ocornut/imgui)
- OpenCL (https://www.khronos.org/opencl/)
- SDL2 (https://libsdl.org/index.php)
- Glad (https://glad.dav1d.de/)
- spdlog (https://github.com/gabime/spdlog)
