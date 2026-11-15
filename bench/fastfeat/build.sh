#!/usr/bin/env bash
# Build the optional C++ featurizer next to this script.
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
if [[ "$(uname)" == "Darwin" ]]; then OUT=libspectre_feat.dylib; else OUT=libspectre_feat.so; fi
"$CXX" -O3 -std=c++17 -shared -fPIC -pthread featurize.cpp -o "$OUT"
echo "built bench/fastfeat/$OUT"
