
mkdir -p "$DEV_BUILD_DIR"
mkdir -p "$DEV_INSTALL_DIR"

# Log
info=" -BUILD- \n"
info+=" - build type = $DEV_BUILD_TYPE\n"
info+=" - root folder = $DEV_DIR\n"
info+=" - install folder = $DEV_INSTALL_DIR\n"
info+=" - build folder = $DEV_BUILD_DIR\n"
info+=" - third_parties folder = $TP_DIR\n"
printf "%b" "$info\n"

if [[ "$CLEAN" -eq "1" ]]; then
    rm -rf "$DEV_BUILD_DIR" "$DEV_INSTALL_DIR"
    mkdir -p "$DEV_BUILD_DIR" "$DEV_INSTALL_DIR"
fi

cd "$DEV_BUILD_DIR"

cmake "$DEV_DIR" \
      -DCMAKE_INSTALL_PREFIX="$DEV_INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=$DEV_BUILD_TYPE

cmake --build "$DEV_BUILD_DIR" --config "$DEV_BUILD_TYPE" --target "install"

cd "$DEV_DIR"