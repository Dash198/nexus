// Remove the #define FUSE_USE_VERSION 31 here (we moved it to CMake)

// Use the explicit path to avoid confusion with v2
#include <fuse3/fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <iostream>

static void* nexus_init(struct fuse_conn_info *conn, struct fuse_config *cfg) {
    (void) conn;
    cfg->kernel_cache = 1;
    std::cout << "[NEXUS] Filesystem initialized!" << std::endl;
    return NULL;
}

static struct fuse_operations nexus_oper = {
    .init = nexus_init,
};

int main(int argc, char *argv[]) {
    // FIX: The second argument must be 'argv', not 'argc'
    return fuse_main(argc, argv, &nexus_oper, NULL);
}
