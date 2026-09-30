#!/bin/bash
# Build script for FlagGems Chinese documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/_build"

echo "Building FlagGems Chinese documentation..."

# Clean previous build
if [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

# Build HTML
echo "Building HTML documentation..."
sphinx-build -b html . "${BUILD_DIR}/html"

echo "Build complete! Output at: ${BUILD_DIR}/html"