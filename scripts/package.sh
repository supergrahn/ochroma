#!/bin/bash
set -e
VERSION="0.1.0"
PACKAGE="ochroma-${VERSION}-linux-x64"

echo "Packaging Ochroma Engine v${VERSION}..."

# Build release
./scripts/build_release.sh

# Create package directory
rm -rf "dist/${PACKAGE}"
mkdir -p "dist/${PACKAGE}/bin"
mkdir -p "dist/${PACKAGE}/assets"
mkdir -p "dist/${PACKAGE}/examples"
mkdir -p "dist/${PACKAGE}/docs"

# Copy binaries
cp target/release/ochroma "dist/${PACKAGE}/bin/"
cp target/release/walking_sim "dist/${PACKAGE}/bin/"

# Copy assets and docs
cp -r assets/* "dist/${PACKAGE}/assets/" 2>/dev/null || true
cp scripts/README_RELEASE.md "dist/${PACKAGE}/README.md"
cp docs/getting_started.md "dist/${PACKAGE}/docs/"
cp -r examples/ "dist/${PACKAGE}/examples/" 2>/dev/null || true

# Create archive
cd dist
tar czf "${PACKAGE}.tar.gz" "${PACKAGE}/"
echo ""
echo "Package created: dist/${PACKAGE}.tar.gz"
ls -lh "${PACKAGE}.tar.gz"
echo ""
echo "Contents:"
tar tzf "${PACKAGE}.tar.gz" | head -20
