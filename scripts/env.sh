set -euo pipefail

DEV_DIR=$(pwd)
DEV_BUILD_DIR="$DEV_DIR/CMakeBuild"
DEV_INSTALL_DIR="$DEV_DIR/install"
DEV_BUILD_TYPE=Release

export DEV_DIR
export DEV_BUILD_DIR
export DEV_INSTALL_DIR
export DEV_BUILD_TYPE
