from conans import ConanFile, tools, CMake


class Conanfile(ConanFile):
    name = "Boids"
    version = "0.1"
    requires = ["sdl/[>=2.0.12]",
                "glad/[>=0.1.29]",
                "opencl-headers/[>=2021.04.29]",
                "opencl-icd-loader/[>=2021.04.29]",
                "spdlog/[>=1.4.1]",
                "imgui/[>=1.79]"
                ]
    settings = "os", "compiler", "arch", "build_type"
    exports = "*"
    generators = "cmake"
    build_policy = "missing"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
