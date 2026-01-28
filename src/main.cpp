// Remove the #define FUSE_USE_VERSION 31 here (we moved it to CMake)

// Use the explicit path to avoid confusion with v2
#include <fcntl.h>
#include <fuse3/fuse.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <dirent.h>

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

int nexus_readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi, enum fuse_readdir_flags flags){
    /*
     * Read the contents of a directory
     *
     * Args:
     *  path: Path to the directory we want to read (relative to filesystem)
     *  buf: Buffer to write to, written by filler
     *  filler: The helper function which conveniently fills up the buffer for us
     *  offset: Offset of the next directory entry, needed for multiple calls to readdir in complex filesystems
     *  fi: Pointer with low-level details about this specific "open" action, ignore it for now.
     *  flags: Any extra instructions from the kernel, ignore for now.
     */

    // Setup the source path
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source+path;

    // Open the directory
    DIR *pDir = opendir(final_path.c_str());

    // Error handling
    if(pDir == nullptr){
        // Return negative errno to the kernel, interpreted automatically
        return -errno;
    }

    // Read loop
    struct dirent *entry;
    // Keep readung until NULL (end of directory)
    while((entry = readdir(pDir))!=NULL){

        // We usually want . and .. for proper navigation
        // If they are filtered out, then 'cd ..' may not work as expected
        // So let us pass everything through for now

        // Call the filler
        // buf: buffer to fill
        // entry->d_name: simple filename (eg: 'image.png')
        // NULL: We let the kernel see stats for later
        // 0: Offset (default)

        filler(buf, entry->d_name, NULL, 0, (fuse_fill_dir_flags)0);
    }

    // Cleanup and return success
    closedir(pDir);
    return 0;
}

int nexus_open(const char *path, struct fuse_file_info *fi){
    /*
     * Attaches a file descriptor to the specified file.
     *
     * Args:
     *  path: The path to the directory
     *  fi: File info struct, used here :)
     */

    // Set up the source
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // Attempt to open and attach a file descriptor to the directory in read-only mode (writing to be handled later)
    int fd = open(final_path.c_str(), O_RDONLY);

    // Error handling
    if(fd==-1){
        // Return the error to the kernel
        return -errno;
    }

    // Pass the file descriptor to fi, to use in the read() method, and return.
    fi->fh = fd;
    return 0;
}

int nexus_read(const char *path, char *buff, size_t size, off_t offset, struct fuse_file_info *fi){
    /*
     * Attempt to read a given file
     *
     * Args:
     *  path: Path to the file
     *  buff: Buffer to write to
     *  size: Number of bytes to read
     *  offset: Offset to start reading from
     *  fi: File info struct, contains attached file descriptor
     */

    // Extract the file descriptor
    int fd = fi->fh;

    // Handle invalid descriptor
    if(fd==-1){
        return -errno;
    }

    // Read bytes from the file
    int res = pread(fd, buff, size, offset);

    // Error handling
    if(res==-1){
        res = -errno;
    }

    // Return res
    return res;
}

int nexus_release(const char *path, struct fuse_file_info *fi){
    /*
     * Called when user is completely done with the file
     *
     * Args:
     *  path: Path to the file
     *  fi: Info about the file
     */

     // Acknowledge the path but do not use it
    (void) path;
    // Close the file descriptor we opened
    close(fi->fh);

    return 0;
}

// The "Employee Handbook" - Defines what function to run for each request and what all we are capable of.
static struct fuse_operations nexus_oper = {
    .getattr = nexus_getattr,
    .open = nexus_open,
    .read = nexus_read,
    .release = nexus_release,
    .readdir = nexus_readdir,
    .init = nexus_init,
};

int main(int argc, char *argv[]) {
    // FIX: The second argument must be 'argv', not 'argc'
    return fuse_main(argc, argv, &nexus_oper, NULL);
}
