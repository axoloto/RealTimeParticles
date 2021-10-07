set -euo pipefail

DEV_DIR=$(pwd)
DEV_BUILD_DIR="$DEV_DIR/CMakeBuild"
DEV_INSTALL_DIR="$DEV_DIR/install"
DEV_PACKAGE=0

if [[ -z "$1" ]] 
then
    echo "No build specified, using Release by default"
    DEV_BUILD_TYPE="Release"
elif [[ ( "$1" != "Release" && "$1" != "Debug" ) ]]
then
    echo "$1 is not a supported build type, using Release instead"
    DEV_BUILD_TYPE="Release"
else
    DEV_BUILD_TYPE="$1"
fi

export DEV_DIR
export DEV_BUILD_DIR
export DEV_INSTALL_DIR
export DEV_BUILD_TYPE
export DEV_PACKAGE
