#!/bin/bash

set -euo pipefail

if [[ "$OSTYPE" == "win32" ]]; then
    ./install/"$DEV_BUILD_TYPE"/bin/RealTimeParticles.exe
else
    ./install/"$DEV_BUILD_TYPE"/bin/RealTimeParticles
fi