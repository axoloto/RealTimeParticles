
CLEAN=0
DEV_BUILD_DIR="$DEV_DIR/CMakeBuild"
DEV_INSTALL_DIR="$DEV_DIR/install"
DEV_BUILD_TYPE=Debug
TP_DIR="$DEV_DIR/third_parties"
USE_OPENCL=true

export CLEAN
export DEV_DIR
export DEV_BUILD_DIR
export DEV_INSTALL_DIR
export DEV_BUILD_TYPE
export TP_DIR
export USE_OPENCL

printf "======================= Starting Conan third-parties installation ========================== \n"

(cd "$DEV_BUILD_DIR" && conan install -s build_type=$DEV_BUILD_TYPE ..)

printf "======================= Finishing Conan third-parties installation ========================== \n"
