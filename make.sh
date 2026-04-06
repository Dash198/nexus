#!/bin/bash

# Define your mount point
MOUNT_POINT="/tmp/nexus"

echo "[1] Cleaning up old mounts and builds..."
# Unmount if it's already mounted (the || true prevents the script from crashing if it wasn't mounted)
fusermount3 -u $MOUNT_POINT 2>/dev/null || true

# Wipe the old build
rm -rf build

echo "[2] Building NEXUS..."
mkdir build && cd build
cmake ..
make -j$(nproc) # Use all CPU cores to compile faster!

echo "[3] Preparing Mount Point..."
# Go back to the root directory
cd ..
mkdir -p $MOUNT_POINT

echo "[4] Booting NEXUS Daemon..."
# Run the daemon in the foreground (-f) so you can see your std::cout AI boot logs
./build/nexus -f $MOUNT_POINT
