from conans import ConanFile, tools, CMake

class Conanfile(ConanFile):
    name = "RealTimeParticles"
    version = "1.0.1"
    requires = ["sdl/[==2.24]",
                "glad/[==0.1.36]",
                "opencl-headers/[==2022.09.30]",
                "opencl-icd-loader/[==2022.09.30]", # might need to be disabled for macos
                "spdlog/[==1.10.0]",
                "imgui/[==1.89]"
                ]
    settings = "os", "compiler", "arch", "build_type"
    exports = "*"
    generators = "cmake_find_package_multi"
    build_policy = "missing"