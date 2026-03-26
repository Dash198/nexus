// Remove the #define FUSE_USE_VERSION 31 here (we moved it to CMake)

// Use the explicit path to avoid confusion with v2
#include <fcntl.h>
#include <fuse3/fuse.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <errno.h>
#include <iostream>
#include <dirent.h>
#include <cstring>
#include <unordered_map>
#include <ctime>
#include <onnxruntime_cxx_api.h>
#include "embedder.hpp"

// Global Map of some paths :)
std::unordered_map<std::string, std::string> global_map = {
    {"/hello", "hello from nexus :)\n"},
    {"/status", "NEXUS Core: ONLINE\n"},
    {"/time", "actually time size will come from our cache, this is just a placeholder :)"}
};

ModelEngine* global_engine = nullptr;

// Dynamic cache, stores ghost directories to generate search results for (only cat for now)
std::unordered_map<std::string, std::string> dynamic_cache;

// Helper function to generate system time.
std::string generate_time_string(){
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime); // This will automatically grab your local IST timezone

    // Format the time and add the crucial newline character
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S\n", timeinfo);

    return std::string(buffer);
}

// Generates search results
std::string generate_search_result(std::string query){
    // Dummy placeholder for now
    return "Simulated results for query " + query + "\n";
}

// Run once, when the filesystem is mounted.
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

    global_engine = new ModelEngine("/home/devansh/repos/nexus/models/all-MiniLM-L6-v2.onnx");
    std::cout << "[NEXUS] AI Engine Booted!" << std::endl;

    return NULL;
}

int nexus_getattr(const char *path, struct stat *stbuf, struct fuse_file_info *fi){
    /*
     * Get attributes about a file.
     * Now.. we make it accept ghost or non-existent directories.
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

     // We'll directly fill the stbuf for global mapped directories.
     if(global_map.find(std::string(path))!=global_map.end()){
        stbuf->st_mode = S_IFREG | 0444;
        stbuf->st_nlink = 1;
        stbuf->st_size = global_map[std::string(path)].length();
        stbuf->st_uid = getuid();
        stbuf->st_gid = getgid();
        time_t now = time(NULL);
        stbuf->st_atime = now;
        stbuf->st_mtime = now;
        stbuf->st_ctime = now;
        if(std::string(path) == "/time"){
            dynamic_cache[path] = generate_time_string();
            stbuf->st_size = dynamic_cache[path].length();
        }
        return 0;
     }

     std::string path_str = std::string(path);

     // We'll check if the user is searching (non-existent directory) and fill the stbuf
     if (path_str == "/search") {
         stbuf->st_mode = S_IFDIR | 0755; // S_IFDIR means "I am a directory!"
         stbuf->st_nlink = 2;             // Directories usually have 2 links
         stbuf->st_uid = getuid();
         stbuf->st_gid = getgid();
         time_t now = time(NULL);
         stbuf->st_atime = now;
         stbuf->st_mtime = now;
         stbuf->st_ctime = now;
         return 0;
     }
     // For an actual search we generate the search result and fill the stbuf.
     if(path_str.starts_with("/search/") && path_str.length()>8){
         stbuf->st_mode = S_IFREG | 0444;
         stbuf->st_nlink = 1;
         stbuf->st_uid = getuid();
         stbuf->st_gid = getgid();
         time_t now = time(NULL);
         stbuf->st_atime = now;
         stbuf->st_mtime = now;
         stbuf->st_ctime = now;
         dynamic_cache[path] = generate_search_result(path_str.substr(8, path_str.length()-8));
         stbuf->st_size = dynamic_cache[path].length();
         return 0;
     }

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
     * Now.. we try to add in ghost directories.
     *
     * Args:
     *  path: Path to the directory we want to read (relative to filesystem)
     *  buf: Buffer to write to, written by filler
     *  filler: The helper function which conveniently fills up the buffer for us
     *  offset: Offset of the next directory entry, needed for multiple calls to readdir in complex filesystems
     *  fi: Pointer with low-level details about this specific "open" action, ignore it for now.
     *  flags: Any extra instructions from the kernel, ignore for now.
     */

    // Special case for root dir
    if(std::string(path) == "/"){
        for (const auto& pair: global_map)
            filler(buf, (pair.first.substr(1, pair.first.length()-1)).c_str(), NULL, 0, (fuse_fill_dir_flags)0);
    }
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


    // Let us now try to intercept any global mapped directories
    if(global_map.find(std::string(path))!=global_map.end()){
        // Just let the kernel pass
        return 0;
    }

    // Similarly for anything tagged search
    if(std::string(path).starts_with("/search/") && std::string(path).size()>8){
        return 0;
    }
    // Set up the source
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // Attempt to open and attach a file descriptor to the directory in read-only mode (writing to be handled later)
    int fd = open(final_path.c_str(), fi->flags);

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

    // Intercepting non-existent directories
    if(global_map.find(std::string(path))!=global_map.end()){
        std::string msg = global_map[std::string(path)];

        if(std::string(path) == "/time"){
            msg = dynamic_cache[path];
        }

        if(offset >= msg.length()){
            return 0;
        }

        size_t bytes = std::min(size,(size_t)(msg.length()-offset));
        memcpy(buff, msg.c_str()+offset, bytes);

        return bytes;
    }

    std::string path_str = std::string(path);

    // Similar procedure for searching.
    if(path_str.starts_with("/search/") && path_str.length()>8){
        std::string msg = dynamic_cache[path];
        if(offset >= msg.length()){
            return 0;
        }

        size_t bytes = std::min(size, (size_t)(msg.length()-offset));
        memcpy(buff, msg.c_str()+offset, bytes);

        return bytes;
    }

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

int nexus_create(const char *path, mode_t mode, struct fuse_file_info *fi){
    /*
     * Attaches a file descriptor to the specified file for writing.
     *
     * Args:
     *  path: Path to the file
     *  mode: Mode to open the descriptor
     *  fi: File info struct
     */

    // Set up the source
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // Attemp to open and attach a file descriptor to the file in write mode
    int fd = open(final_path.c_str(), fi->flags, mode);

    // Error handling
    if(fd == -1){
        return -errno;
    }

    // Pass the descriptor
    fi->fh = fd;
    return 0;
}

int nexus_write(const char *path, const char *buf, size_t size, off_t offset, struct fuse_file_info *fi){
    /*
     * Write to a given file using a file descriptor
     *
     * Args:
     *  path: Path of the file
     *  buf: Buffer to write to
     *  size: Number of bytes to write
     *  offset: Offset to write from
     *  fi: File info struct
     */

    // Get the file descriptor
    int fd = fi->fh;

    // Error handling for invalid fd
    if(fd == -1){
        return -errno;
    }

    // Write using pwrite()
    int res = pwrite(fd, buf, size, offset);

    // Handle invalid writing
    if(res == -1){
        res = -errno;
    }

    return res;
}
int nexus_unlink(const char *path) {
    /*
     * Delete a file from the filesystem
     *
     * Args:
     *  path: Path of the file
     */

    // Get the source.
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // Attempt to delete the file.
    int res = unlink(final_path.c_str());

    // Handle error.
    if (res == -1) {
        return -errno;
    }

    return 0;
}

int nexus_mkdir(const char *path, mode_t mode) {
    /*
     * Create a directory
     *
     * Args:
     *  path: Path of the directory
     *  mode: Creation mode/Permissions
     */

    // Construct the actual path
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // mkdir takes the path and the permissions (mode)
    int res = mkdir(final_path.c_str(), mode);

    // Handle errors
    if (res == -1) {
        return -errno;
    }

    return 0;
}

int nexus_rmdir(const char *path) {
    /*
     * Delete a directory from the filesystem
     *
     * Args:
     *  path: Path to the directory
     */

    // Create the actual path
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // Attempt to remove the directory
    int res = rmdir(final_path.c_str());

    // Handle errors
    if (res == -1) {
        return -errno;
    }

    return 0;
}
// The "Employee Handbook" - Defines what function to run for each request and what all we are capable of.
static struct fuse_operations nexus_oper = {
    .getattr = nexus_getattr,
    .mkdir = nexus_mkdir,
    .unlink = nexus_unlink,
    .rmdir = nexus_rmdir,
    .open = nexus_open,
    .read = nexus_read,
    .write = nexus_write,
    .release = nexus_release,
    .readdir = nexus_readdir,
    .init = nexus_init,
    .create = nexus_create,
};

int main(int argc, char *argv[]) {
    // FIX: The second argument must be 'argv', not 'argc'
    return fuse_main(argc, argv, &nexus_oper, NULL);
}
