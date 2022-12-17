from conans import ConanFile, tools, CMake

class Conanfile(ConanFile):
    name = "RealTimeParticles"
    version = "1.0.1"
    requires = ["sdl/[>=2.0.12]",
                "glad/[>=0.1.29]",
                "opencl-headers/[>=2021.04.29]",
              #  "opencl-icd-loader/[>=2021.04.29]", //temp, not working with Apple
                "spdlog/[>=1.9.2]",
                "imgui/[>=1.85]"
                ]
    settings = "os", "compiler", "arch", "build_type"
    exports = "*"
    generators = "cmake"
    build_policy = "missing"