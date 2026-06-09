#!/usr/bin/env bash
# Rebuild voro_wasm.js from the voro++ source (voro_src/) + voro_wrap.cpp.
# Requires Emscripten on PATH (e.g. `source ~/emsdk/emsdk_env.sh`).
set -e
emcc -O3 -I voro_src/src \
  voro_src/src/cell.cc voro_src/src/common.cc voro_src/src/container.cc voro_src/src/container_prd.cc \
  voro_src/src/unitcell.cc voro_src/src/v_compute.cc voro_src/src/c_loops.cc voro_src/src/wall.cc \
  voro_src/src/pre_container.cc voro_src/src/v_base.cc voro_wrap.cpp \
  -lembind -s MODULARIZE=1 -s EXPORT_ES6=1 -s ENVIRONMENT=web,node \
  -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 -o voro_wasm.js
echo "built voro_wasm.js"
