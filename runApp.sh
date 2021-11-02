#!/bin/bash

source ./scripts/env.sh "$1"
./scripts/build.sh
./scripts/pack.sh
./scripts/run.sh