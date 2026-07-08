#include <algorithm>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <fuse3/fuse.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <pthread.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "db/vector_store.hpp"
#include "embedder.hpp"
#include "image_parser.hpp"
#include "router.hpp"
#include "search_cache.hpp"
#include "tokenizer/tokenizer.hpp"

struct VecCtxt {
  Tokenizer *tokenizer;
  ModelEngine *engine;
  VectorStore *store;
  SearchCache *cache;

  std::string model_path =
      "/home/devansh/repos/nexus/models/all-MiniLM-L6-v2.onnx";
};

// Get AI context inside of any fuse function
#define VEC_DATA ((VecCtxt *)fuse_get_context()->private_data)

// Global Map of some paths
std::unordered_map<std::string, std::string> global_map = {
    {"/hello", "hello from vecfs :)\n"},
    {"/status", "VecFS Core: ONLINE\n"},
    {"/time", "actually time size will come from our cache, this is just a "
              "placeholder"}};

// Helper function to generate system time.
std::string generate_time_string() {
  time_t rawtime;
  struct tm *timeinfo; // tm struct breaks time into human readable ints
  char buffer[80];     // Final string that will store time

  // Obtain the raw UNIX timestamp and calculate the exact Year, Month,
  // Day, Hour, Minute, Second, and adjust it to the local timezone.
  time(&rawtime);
  timeinfo = localtime(&rawtime);

  // Format the time using the strfttime fn
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S\n", timeinfo);

  return std::string(buffer);
}

// Generates search results for user queries
std::string generate_search_result(std::string query) {

  // Load the AI ctxt
  struct VecCtxt *ctxt = VEC_DATA;

  // Format the query
  std::replace(query.begin(), query.end(), '_', ' ');

  // Check if a cached result exists, return it.
  std::string cache_result = ctxt->cache->get(query);
  if (cache_result != "") {
    return cache_result;
  }

  // Open a stringstream to write into
  std::stringstream ss;
  ss << "Search Results:\n\n";

  // Generate the query embedding
  Encoding enc = ctxt->tokenizer->encode(query);
  std::vector<float> embedding = ctxt->engine->generate_embedding(enc);

  // Search the top k (5) results
  std::vector<std::string> results = ctxt->store->search(embedding, 5);

  // Results would only be empty if no files are indexed
  if (results.empty()) {
    ss << "No files indexed in memory\n";
  } else {
    // Print the search results
    for (size_t i = 0; i < results.size(); i++) {
      ss << " " << i + 1 << ". " << results[i] << "\n";
    }
  }

  ss << "\n";

  // Set a similarity threshold that any new file added has to surpass to
  // regenerate the search result If the store is empty, set threshold to -2.0f
  // so ANY new file kills the cache
  float threshold = results.empty() ? -2.0f : 0.0f;

  std::string final_output = ss.str();

  // Add the query-value pair to the cache
  ctxt->cache->put(query, embedding, final_output, results, threshold);
  return final_output;
}

// Go through the given path to identify edited/deleted/new files and update the
// vector store
void sync_drive(std::string path, VecCtxt *ctxt,
                std::unordered_set<std::string> &found_files) {

  // Open the path, return if null
  DIR *dir = opendir(path.c_str());
  if (dir == nullptr)
    return;

  VectorStore *store = ctxt->store;

  struct dirent *entry; // Holds the name and type of whatever is found
  struct stat buf;      // Holds deeper metadata (size, permissions etc.)

  // readdir reads the directory stream one-by-one, returns NULL once all
  // the items in the current dir are over.
  while ((entry = readdir(dir)) != NULL) {
    // Skip . (current dir) and .. (parent dir), becomes an infinite loop
    // otherwise.
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
      continue;

    // DT_REG means regular file (not symlink/folder)
    if (entry->d_type == DT_REG) {
      // Construct the full path
      std::string full_path = path + "/" + entry->d_name;

      // Dump all the metadata of the file into the buf
      lstat(full_path.c_str(), &buf);

      // Strip the base directory to match the ledger
      std::string base_dir = "/home/devansh/repos/nexus/vecfs_data";
      std::string fuse_path = full_path.substr(base_dir.length());

      // Insert the FUSE path, not the absolute path
      found_files.insert(fuse_path);

      // Query the store using the FUSE path
      // Check if it is a new file or edited.
      if (!store->contains(fuse_path) ||
          buf.st_mtime > store->get_mtime(fuse_path)) {
        // If it satiesfies either condition above, then geenrate the embedding
        std::ifstream inFile(full_path);

        std::stringstream buffer;
        buffer << inFile.rdbuf();
        std::string content = buffer.str();

        if (!content.empty()) {
          std::vector<float> embedding;
          enum FileType file_type = detect_file_type(full_path);

          if (file_type == TEXT) {
            Encoding encoding = ctxt->tokenizer->encode(content);
            embedding = ctxt->engine->generate_embedding(encoding);
          } else if (file_type == IMAGE) {
            int w, h;
            std::vector<unsigned char> processed_image =
                process_image(full_path, w, h);
            std::vector<unsigned char> resized_image =
                resize_image(processed_image.data(), w, h);
            std::vector<float> norm_img = normalize_image(resized_image);

            // TODO: Call Embedder
          }
          bool is_mutant = store->contains(fuse_path);

          store->upsert(fuse_path, embedding);

          // If the file is edited then replace the embedding in cache if exists
          if (is_mutant) {
            ctxt->cache->invalidate_on_edit(fuse_path, embedding);
          }
        }
      }
    }
    // If it is a directory then recursively sync
    else if (entry->d_type == DT_DIR) {
      sync_drive(path + "/" + entry->d_name, ctxt, found_files);
    }
  }
  closedir(dir);
}

// Remove any deleted files from the saved store
void remove_invalid(VecCtxt *ctxt,
                    std::unordered_set<std::string> &found_files) {
  // Get all the paths in the store
  std::vector<std::string> ctxt_files = ctxt->store->get_all_paths();

  // Check if the paths exist, delete otherwise from both store and cache
  for (const auto &path : ctxt_files) {
    if (!found_files.contains(path)) {
      ctxt->store->remove(path);
      ctxt->cache->invalidate_on_delete(path);
    }
  }
}

// Background Thread function that syncs the drive
void background_sweeper(VecCtxt *ctxt) {
  std::unordered_set<std::string> found_files;
  sync_drive("/home/devansh/repos/nexus/vecfs_data", ctxt, found_files);
  remove_invalid(ctxt, found_files);
}

// Run once, when the filesystem is mounted.
static void *vec_init(struct fuse_conn_info *conn, struct fuse_config *cfg) {
  /*
   * conn: connection information
   * cfg: configuration flags
   */

  // Accept conn but do not use it
  (void)conn;

  // Enabling the kernel cache
  cfg->kernel_cache = 1;
  std::cout << "[VecFS] Filesystem initialized!" << std::endl;

  // Initialize the AI components
  struct VecCtxt *ctxt = VEC_DATA;
  ctxt->tokenizer = new Tokenizer();
  ctxt->engine = new ModelEngine(ctxt->model_path);
  ctxt->store = new VectorStore();
  ctxt->store->load_from_disk();
  ctxt->cache = new SearchCache(50);

  std::cout << "[VecFS] AI Engine Booted!" << std::endl;

  // Detach a thread to sweep the mount point to identify any new files,
  // edited files or deleted ones and update the vector store
  std::thread sweeper(background_sweeper, ctxt);
  sweeper.detach();

  return VEC_DATA;
}

// Get metadata about a file/directory
int vec_getattr(const char *path, struct stat *stbuf,
                struct fuse_file_info *fi) {
  /*
   * Get attributes about a file.
   * Can also accept ghost or non-existent directories.
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

  // We'll directly fill the stbuf for global mapped directories that
  // do not actually exist on the disk
  if (global_map.find(std::string(path)) != global_map.end()) {
    stbuf->st_mode =
        S_IFREG |
        0444; // IFREG -> regular file, 0444 -> Strictly Read-Only for everyone
    stbuf->st_nlink = 1;
    stbuf->st_size = global_map[std::string(path)].length();
    stbuf->st_uid = getuid();
    stbuf->st_gid = getgid();
    time_t now = time(NULL);
    stbuf->st_atime = now;
    stbuf->st_mtime = now;
    stbuf->st_ctime = now;
    if (std::string(path) == "/time") {
      stbuf->st_size = generate_time_string().length();
    }
    return 0;
  }

  std::string path_str = std::string(path);

  // We'll check if the user is searching (non-existent directory) and fill the
  // stbuf
  if (path_str == "/search") {
    stbuf->st_mode = S_IFDIR | 0755; // S_IFDIR -> directory
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
  if (path_str.starts_with("/search/") && path_str.length() > 8) {
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
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  // Get actual path of file
  std::string final_path = source + path;

  // Call the linux function lstat to retrieve information about the file.
  // c_str() is a function that converts strings in C++ to compatible C char
  // arrays stbuf is automatically filled
  int status = lstat(final_path.c_str(), stbuf);

  if (status == -1) {
    // lstat failed. Reason is stored in global 'errno' variable.
    // FUSE expects us to return the errno as a negative number.
    return -errno;
  } else {
    // Success
    return 0;
  }
}

// Read a directory and get all files in it
int vec_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                off_t offset, struct fuse_file_info *fi,
                enum fuse_readdir_flags flags) {
  /*
   * Read the contents of a directory
   * Also add in ghost directories.
   *
   * Args:
   *  path: Path to the directory we want to read (relative to filesystem)
   *  buf: Buffer to write to, written by filler
   *  filler: The helper function which conveniently fills up the buffer for us
   *  offset: Offset of the next directory entry, needed for multiple calls to
   * readdir in complex filesystems fi: Pointer with low-level details about
   * this specific "open" action, ignore it for now. flags: Any extra
   * instructions from the kernel, ignore for now.
   */

  // Special case for root dir, add in the global virtual files
  if (std::string(path) == "/") {
    for (const auto &pair : global_map)
      filler(buf, (pair.first.substr(1, pair.first.length() - 1)).c_str(), NULL,
             0, (fuse_fill_dir_flags)0);
  }
  // Setup the source path
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  std::string final_path = source + path;

  // Open the directory
  DIR *pDir = opendir(final_path.c_str());

  // Error handling
  if (pDir == nullptr) {
    // Return negative errno to the kernel, interpreted automatically
    return -errno;
  }

  // Read loop
  struct dirent *entry;
  // Keep reading until NULL (end of directory)
  while ((entry = readdir(pDir)) != NULL) {

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

int vec_open(const char *path, struct fuse_file_info *fi) {
  /*
   * Attaches a file descriptor to the specified file.
   *
   * Args:
   *  path: The path to the directory
   *  fi: File info struct, used here :)
   */

  // Let us now try to intercept any global mapped directories
  if (global_map.find(std::string(path)) != global_map.end()) {
    // Just let the kernel pass
    return 0;
  }

  // Similarly for anything tagged search
  if (std::string(path).starts_with("/search/") &&
      std::string(path).size() > 8) {
    return 0;
  }
  // Set up the source
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  std::string final_path = source + path;

  // Attempt to open and attach a file descriptor to the directory in read-only
  // mode (writing to be handled later)
  int fd = open(final_path.c_str(), fi->flags);

  // Error handling
  if (fd == -1) {
    // Return the error to the kernel
    return -errno;
  }

  // Pass the file descriptor to fi, to use in the read() method, and return.
  fi->fh = fd;
  return 0;
}

int vec_read(const char *path, char *buff, size_t size, off_t offset,
             struct fuse_file_info *fi) {
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
  if (global_map.find(std::string(path)) != global_map.end()) {
    std::string msg = global_map[std::string(path)];

    if (std::string(path) == "/time") {
      msg = generate_time_string();
    }

    if (offset >= msg.length()) {
      return 0;
    }

    size_t bytes = std::min(size, (size_t)(msg.length() - offset));
    memcpy(buff, msg.c_str() + offset, bytes);

    return bytes;
  }

  std::string path_str = std::string(path);

  // Similar procedure for searching.
  if (path_str.starts_with("/search/") && path_str.length() > 8) {
    std::string msg = generate_search_result(path_str.substr(8));
    if (offset >= msg.length()) {
      return 0;
    }

    size_t bytes = std::min(size, (size_t)(msg.length() - offset));
    memcpy(buff, msg.c_str() + offset, bytes);

    return bytes;
  }

  // Extract the file descriptor
  int fd = fi->fh;

  // Handle invalid descriptor
  if (fd == -1) {
    return -errno;
  }

  // Read bytes from the file
  int res = pread(fd, buff, size, offset);

  // Error handling
  if (res == -1) {
    res = -errno;
  }

  // Return res
  return res;
}

int vec_release(const char *path, struct fuse_file_info *fi) {
  /*
   * Called when user is completely done with the file
   *
   * Args:
   *  path: Path to the file
   *  fi: Info about the file
   */

  if (fi->fh != -1) {
    close(fi->fh);
  }
  struct VecCtxt *ctxt = VEC_DATA;

  std::string final_path =
      "/home/devansh/repos/nexus/vecfs_data" + std::string(path);
  std::ifstream inFile(final_path);
  if (!inFile.is_open())
    return 0;

  std::stringstream buffer;
  buffer << inFile.rdbuf();
  std::string content = buffer.str();

  if (!content.empty()) {
    std::vector<float> embedding;
    enum FileType file_type = detect_file_type(full_path);

    if (file_type == TEXT) {
      Encoding encoding = ctxt->tokenizer->encode(content);
      embedding = ctxt->engine->generate_embedding(encoding);
    } else if (file_type == IMAGE) {
      int w, h;
      std::vector<unsigned char> processed_image =
          process_image(full_path, w, h);
      std::vector<unsigned char> resized_image =
          resize_image(processed_image.data(), w, h);
      std::vector<float> norm_img = normalize_image(resized_image);

      // TODO: Call Embedder
    }
    ctxt->store->upsert(path, embedding);

    ctxt->cache->invalidate_on_edit(path, embedding);
  }

  inFile.close();
  return 0;
}

int vec_create(const char *path, mode_t mode, struct fuse_file_info *fi) {
  /*
   * Attaches a file descriptor to the specified file for writing.
   *
   * Args:
   *  path: Path to the file
   *  mode: Mode to open the descriptor
   *  fi: File info struct
   */

  // Set up the source
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  std::string final_path = source + path;

  // Attemp to open and attach a file descriptor to the file in write mode
  int fd = open(final_path.c_str(), fi->flags, mode);

  // Error handling
  if (fd == -1) {
    return -errno;
  }

  // Pass the descriptor
  fi->fh = fd;
  return 0;
}

int vec_write(const char *path, const char *buf, size_t size, off_t offset,
              struct fuse_file_info *fi) {
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
  if (fd == -1) {
    return -errno;
  }

  // Write using pwrite()
  int res = pwrite(fd, buf, size, offset);

  // Handle invalid writing
  if (res == -1) {
    res = -errno;
  }

  return res;
}

int vec_unlink(const char *path) {
  /*
   * Delete a file from the filesystem
   *
   * Args:
   *  path: Path of the file
   */

  struct VecCtxt *ctxt = VEC_DATA;

  // Get the source.
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  std::string final_path = source + path;

  // Attempt to delete the file.
  int res = unlink(final_path.c_str());

  // Handle error.
  if (res == -1) {
    return -errno;
  }

  // Remove the path from the vector store
  ctxt->store->remove(path);
  ctxt->cache->invalidate_on_delete(path);

  return 0;
}

int vec_mkdir(const char *path, mode_t mode) {
  /*
   * Create a directory
   *
   * Args:
   *  path: Path of the directory
   *  mode: Creation mode/Permissions
   */

  // Construct the actual path
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  std::string final_path = source + path;

  // mkdir takes the path and the permissions (mode)
  int res = mkdir(final_path.c_str(), mode);

  // Handle errors
  if (res == -1) {
    return -errno;
  }

  return 0;
}

int vec_rmdir(const char *path) {
  /*
   * Delete a directory from the filesystem
   *
   * Args:
   *  path: Path to the directory
   */

  // Create the actual path
  std::string source = "/home/devansh/repos/nexus/vecfs_data";
  std::string final_path = source + path;

  // Attempt to remove the directory
  int res = rmdir(final_path.c_str());

  // Handle errors
  if (res == -1) {
    return -errno;
  }

  return 0;
}

// Creates a mapping for what function to run for each request made by the FS
static struct fuse_operations vec_oper = {
    .getattr = vec_getattr,
    .mkdir = vec_mkdir,
    .unlink = vec_unlink,
    .rmdir = vec_rmdir,
    .open = vec_open,
    .read = vec_read,
    .write = vec_write,
    .release = vec_release,
    .readdir = vec_readdir,
    .init = vec_init,
    .create = vec_create,
};

int main(int argc, char *argv[]) {
  // Starting up the daemon.
  std::cout << "[VecFS] Booting Daemon.." << std::endl;

  // Allocate context on the heap
  VecCtxt *ai_ctxt = new VecCtxt();

  // We don't initialize the AI here, we just crete empty pointers
  ai_ctxt->engine = nullptr;
  ai_ctxt->tokenizer = nullptr;
  ai_ctxt->store = nullptr;
  ai_ctxt->cache = nullptr;

  // fuse_main(argc, argv, &operations_struct, PRIVATE_DATA_POINTER)
  int fuse_stat = fuse_main(argc, argv, &vec_oper, ai_ctxt);

  // Cleanup when fuse unmounts

  // Save Vector Store to disk
  if (ai_ctxt->store != nullptr) {
    ai_ctxt->store->save_to_disk();
  }

  // Delete the pointers
  delete ai_ctxt->engine;
  delete ai_ctxt->store;
  delete ai_ctxt->tokenizer;
  delete ai_ctxt->cache;
  delete ai_ctxt;

  return fuse_stat;
}
