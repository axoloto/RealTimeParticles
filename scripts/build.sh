#!/bin/bash

set -euo pipefail

printf "========================= START $DEV_BUILD_TYPE BUILD ============================ \n"

paths=" - root folder = $DEV_DIR\n"
paths+=" - build folder = $DEV_BUILD_DIR\n"
paths+=" - install folder = $DEV_INSTALL_DIR\n"
printf "%b" "$paths\n"

mkdir -p "$DEV_BUILD_DIR/$DEV_BUILD_TYPE" "$DEV_INSTALL_DIR/$DEV_BUILD_TYPE"

cd "$DEV_BUILD_DIR/$DEV_BUILD_TYPE"

printf "========================= START CMAKE ============================ \n"


cmake "$DEV_DIR" \
      -DCMAKE_INSTALL_PREFIX="$DEV_INSTALL_DIR/$DEV_BUILD_TYPE" \
      -DCMAKE_BUILD_TYPE=$DEV_BUILD_TYPE \
      -DCMAKE_BUILD_DIR="$DEV_BUILD_DIR/$DEV_BUILD_TYPE" \

cmake --build "$DEV_BUILD_DIR/$DEV_BUILD_TYPE" --config "$DEV_BUILD_TYPE" --target "install"

printf "========================== END CMAKE ============================= \n"

cd "$DEV_DIR"

printf "========================== END $DEV_BUILD_TYPE BUILD ============================ \n"
