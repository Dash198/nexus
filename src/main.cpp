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
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "embedder.hpp"
#include "tokenizer/tokenizer.hpp"
#include "db/vector_store.hpp"
#include "search_cache.hpp"

struct NexusCtxt{
    Tokenizer* tokenizer;
    ModelEngine* engine;
    VectorStore* store;
    SearchCache* cache;

    std::string model_path = "/home/devansh/repos/nexus/models/all-MiniLM-L6-v2.onnx";
};

// Get AI context inside of any fuse function
#define NEXUS_DATA ((NexusCtxt*) fuse_get_context()->private_data)

// Global Map of some paths :)
std::unordered_map<std::string, std::string> global_map = {
    {"/hello", "hello from nexus :)\n"},
    {"/status", "NEXUS Core: ONLINE\n"},
    {"/time", "actually time size will come from our cache, this is just a placeholder :)"}
};

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

    // Load the AI ctxt
    struct NexusCtxt *ctxt = NEXUS_DATA;

    std::replace(query.begin(), query.end(), '_', ' ');

    std::string cache_result = ctxt->cache->get(query);
    if(cache_result!=""){
     return cache_result;
    }

    std::stringstream ss;
    ss << "Search Results:\n\n";

    Encoding enc = ctxt->tokenizer->encode(query);
    std::vector<float> embedding = ctxt->engine->generate_embedding(enc);

    std::vector<std::string> results = ctxt->store->search(embedding, 5);

    if(results.empty()){
        ss << "No files indexed in memory\n";
    }
    else{
        for(size_t i=0; i<results.size(); i++){
            ss << " " << i+1 << ". " << results[i] << "\n";
        }
    }

    ss << "\n";
    // If the store is empty, set threshold to -2.0f so ANY new file kills the cache
    float threshold = results.empty() ? -2.0f : 0.0f;

    std::string final_output = ss.str();
    ctxt->cache->put(query, embedding, final_output, results, threshold);
    return final_output;
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

    // Initialize the AI components
    struct NexusCtxt *ctxt = NEXUS_DATA;
    ctxt->tokenizer = new Tokenizer();
    ctxt->engine = new ModelEngine(ctxt->model_path);
    ctxt->store = new VectorStore();
    ctxt->store->load_from_disk();
    ctxt->cache = new SearchCache(50);

    std::cout << "[NEXUS] AI Engine Booted!" << std::endl;

    return NEXUS_DATA;
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
            stbuf->st_size = generate_time_string().length();
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
         std::string query = path_str.substr(8);
         stbuf->st_size = generate_search_result(query).length();
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
            msg = generate_time_string();
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
        std::string msg = generate_search_result(path_str.substr(8));
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

    if(fi->fh!=-1){
        close(fi->fh);
    }
    struct NexusCtxt *ctxt = NEXUS_DATA;

    std::string final_path = "/home/devansh/repos/nexus/nexus_data" + std::string(path);
    std::ifstream inFile(final_path);
    if (!inFile.is_open()) return 0;

    std::stringstream buffer;
    buffer << inFile.rdbuf();
    std::string content = buffer.str();

    if (!content.empty()) {
        Encoding encoding = ctxt->tokenizer->encode(content);
        std::vector<float> embedding = ctxt->engine->generate_embedding(encoding);
        ctxt->store->upsert(path, embedding);
    }
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

    struct NexusCtxt *ctxt = NEXUS_DATA;

    // Get the source.
    std::string source = "/home/devansh/repos/nexus/nexus_data";
    std::string final_path = source + path;

    // Attempt to delete the file.
    int res = unlink(final_path.c_str());

    // Handle error.
    if (res == -1) {
        return -errno;
    }

    // Remove the path from the vector store
    ctxt->store->remove(path);

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
    std::cout << "[NEXUS] Booting Daemon.." << std::endl;

    // Allocate context on the heap
    NexusCtxt* ai_ctxt = new NexusCtxt();

    // We don't initialize the AI here, we just crete empty pointers
    ai_ctxt->engine = nullptr;
    ai_ctxt->tokenizer = nullptr;
    ai_ctxt->store = nullptr;
    ai_ctxt->cache = nullptr;

    // fuse_main(argc, argv, &operations_struct, PRIVATE_DATA_POINTER)
    int fuse_stat = fuse_main(argc, argv, &nexus_oper, ai_ctxt);

    // Cleanup when fuse unmounts
    delete ai_ctxt->engine;
    delete ai_ctxt->store;
    delete ai_ctxt->tokenizer;
    delete ai_ctxt->cache;
    delete ai_ctxt;

    return fuse_stat;
}
