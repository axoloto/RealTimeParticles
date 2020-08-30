# Log
info=" -BUILD- \n"
info+=" - build type = $DEV_BUILD_TYPE\n"
info+=" - root folder = $DEV_DIR\n"
info+=" - install folder = $DEV_INSTALL_DIR\n"
info+=" - build folder = $DEV_BUILD_DIR\n"
info+=" - third_parties folder = $TP_DIR\n"
info+=" - openCL = $USE_OPENCL\n"
printf "%b" "$info\n"

if [[ "$CLEAN" -eq "1" ]]; then
    rm -rf "$DEV_BUILD_DIR" "$DEV_INSTALL_DIR"
    mkdir -p "$DEV_BUILD_DIR" "$DEV_INSTALL_DIR"
fi

cd "$DEV_BUILD_DIR"

printf "======================= Starting Conan third-parties installation ========================== \n"

#conan install -s build_type=$DEV_BUILD_TYPE ..
conan install -s build_type=$DEV_BUILD_TYPE .. -s compiler.runtime=MDd --build=missing

printf "======================= Finishing Conan third-parties installation ========================== \n"

cmake "$DEV_DIR" \
      -DCMAKE_INSTALL_PREFIX="$DEV_INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=$DEV_BUILD_TYPE \
      -DCMAKE_BUILD_DIR=$DEV_BUILD_DIR \
      -DCMAKE_GENERATOR_PLATFORM=x64 \
      -DUSE_OPENCL=$USE_OPENCL

cmake --build "$DEV_BUILD_DIR" --config "$DEV_BUILD_TYPE" --target "install"

cd "$DEV_DIR"