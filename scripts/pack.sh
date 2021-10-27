#!/bin/bash

set -euo pipefail

if [[ $DEV_PACKAGE == 0 ]] 
then
    echo "No package desired, skipping packaging"
    exit 0
fi

printf "========================= START $DEV_BUILD_TYPE PACKAGE ============================ \n"

paths+=" - install folder = $DEV_INSTALL_DIR\n"
paths+=" - package folder = $DEV_INSTALL_DIR\n"
printf "%b" "$paths\n"

cd "$DEV_BUILD_DIR/$DEV_BUILD_TYPE"

printf "========================= START CPACK ============================ \n"

cpack -C $DEV_BUILD_TYPE

printf "========================== END CPACK ============================= \n"

cd "$DEV_DIR"

printf "========================== END $DEV_BUILD_TYPE PACKAGE ============================ \n"
