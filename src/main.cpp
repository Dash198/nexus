// Remove the #define FUSE_USE_VERSION 31 here (we moved it to CMake)

// Use the explicit path to avoid confusion with v2
#include <fuse3/fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <iostream>

// Run once, when the fileesystem is mounted.
static void* nexus_init(struct fuse_conn_info *conn, struct fuse_config *cfg) {
    /*
     * conn: connection information
     * cfg: configuration flags
     */

    // Accept conn but do not use it
    (void) conn;
    // telling the Kernel: "If a user reads a file, you can keep a copy in RAM. Don't call me again for the same file 1 millisecond later."
    cfg->kernel_cache = 1;
    std::cout << "[NEXUS] Filesystem initialized!" << std::endl;
    return NULL;
}

int nexus_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi){
    /*
     * Get attributes about a file.
     *
     * Args:
     *  path: Path the user is asking about, relative to the mount point.
     *  stbuf: Pointer to an empty struct stat, which needs to be filled
     *  fi: File info, ignore for now
     *
     * Return:
     *  0: Success
     *  -ENOENT: Negative Error, No Entry (Failure)
     */

     // Source directory to mirror
     std::string source = "/home/devansh/repos/nexus/nexus_data";
     // Get actual path of file
     std::string final_path = source+path;

     // Call the linux function lstat to retrieve information about the file.
     // c_str() is a function that converts strings in C++ to compatible C char arrays
     // stbuf is automatically filled
     int status = lstat(final_path.c_str(), stbuf);

     if(status==-1){
         // lstat failed. Reason is stored in global 'errno' variable.
         // FUSE expects us to return the errno as a negative number.
         return -errno;
     }
     else{
         // Success
         return 0;
     }
}

// The "Employee Handbook" - Defines what function to run for each request and what all we are capable of.
static struct fuse_operations nexus_oper = {
    .getattr = nexus_getattr,
    .init = nexus_init,
};

int main(int argc, char *argv[]) {
    // FIX: The second argument must be 'argv', not 'argc'
    return fuse_main(argc, argv, &nexus_oper, NULL);
}
