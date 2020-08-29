from conans import ConanFile, tools, CMake


class Conanfile(ConanFile):
    name = "Boids"
    version = "0.1"
    requires = ["sdl2/2.0.12@bincrafters/stable",
                "glad/0.1.29@bincrafters/stable",
                "boost_build/1.69.0@bincrafters/stable"]
    settings = "os", "compiler", "arch", "build_type"
    exports = "*"
    generators = "cmake"
    build_policy = "missing"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
