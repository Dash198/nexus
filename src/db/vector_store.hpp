#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../hnswlib/hnswlib.h"

// Embedding struct for syncing purposes
typedef struct {
  std::vector<float> embedding;
  time_t mtime;
  int id;
} embedding_data;

class VectorStore {
private:
  // Cache of the vector store
  std::unordered_map<std::string, embedding_data> cache;

  int current_id = 0;
  std::unordered_map<std::string, int> path_to_id;
  std::unordered_map<int, std::string> id_to_path;

  // Mutex lock for threading
  mutable std::shared_mutex rw_lock;

  hnswlib::InnerProductSpace *space;
  hnswlib::HierarchicalNSW<float> *alg_hnsw;

  int hnsw_dims = 384;

  // Cosine Sim function
  float cosine_similarity(const std::vector<float> &vec_a,
                          const std::vector<float> &vec_b) {
    int n = vec_a.size();

    float dp = 0.0f;
    float n_a = 0.0f, n_b = 0.0f;

    for (int i = 0; i < n; i++) {
      dp += vec_a[i] * vec_b[i];
      n_a += vec_a[i] * vec_a[i];
      n_b += vec_b[i] * vec_b[i];
    }

    if (n_a < 1e-8 || n_b < 1e-8) {
      return 0.0f;
    }

    return (dp / (std::sqrt(n_a) * std::sqrt(n_b)));
  }

public:
  VectorStore() {
    space = new hnswlib::InnerProductSpace(hnsw_dims);

    alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, 10000, 16, 200);
  }

  ~VectorStore() = default;

  // Insert an embedding into the store
  void upsert(const std::string &filepath,
              const std::vector<float> &embedding) {
    std::unique_lock<std::shared_mutex> lock(rw_lock);
    time_t now = time(NULL);

    int id;

    if (path_to_id.find(filepath) != path_to_id.end()) {
      id = path_to_id[filepath];
    } else {
      id = current_id++;
      id_to_path[id] = filepath;
      path_to_id[filepath] = id;
    }
    alg_hnsw->addPoint(embedding.data(), id);
    cache[filepath] = {embedding, now, id};
  }

  // Remove an embedding
  void remove(const std::string &filepath) {
    std::unique_lock<std::shared_mutex> lock(rw_lock);
    cache.erase(filepath);
    alg_hnsw->markDelete(path_to_id[filepath]);
  }

  // Get all the paths in the store
  std::vector<std::string> get_all_paths() {
    std::shared_lock<std::shared_mutex> lock(rw_lock);

    std::vector<std::string> paths;
    for (auto const &pair : cache) {
      paths.push_back(pair.first);
    }

    return paths;
  }

  // Helper function to check if path exists in store
  bool contains(const std::string path) {
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    if (cache.find(path) == cache.end())
      return false;
    return true;
  }

  // Assuming path exists, get last modified time
  time_t get_mtime(const std::string path) {
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    return cache[path].mtime;
  }

  // The actual search engine
  std::vector<std::string> search(const std::vector<float> &query_embedding,
                                  int top_k = 5) {
    std::shared_lock<std::shared_mutex> lock(rw_lock);
    std::priority_queue<std::pair<float, std::string>> pq;

    for (auto &pair : cache) {
      float score = cosine_similarity(query_embedding, pair.second.embedding);
      pq.push({score, pair.first});
    }

    std::vector<std::string> results;
    for (int i = 0; i < top_k && !pq.empty(); i++) {
      // results.push_back(pq.top().second);
      pq.pop();
    }

    auto hnsw_results = alg_hnsw->searchKnn(query_embedding.data(), top_k);

    while (!hnsw_results.empty()) {
      int id = hnsw_results.top().second;
      results.push_back(id_to_path[id]);
      hnsw_results.pop();
    }

    std::reverse(results.begin(), results.end());
    return results;
  }

  // Save all entries to disk
  void save_to_disk() {

    // Acquire a lock
    std::unique_lock<std::shared_mutex> lock(rw_lock);

    // Open the file to write too
    std::string db_path = "/home/devansh/repos/nexus/vector_store.bin";
    std::ofstream out_file(db_path, std::ios::out | std::ios::binary);
    if (!out_file.is_open()) {
      std::cerr << "[VecFS] Error opening file for writing\n";
    }

    // Write the number of items
    size_t cache_size = cache.size();
    out_file.write(reinterpret_cast<const char *>(&cache_size), sizeof(size_t));

    // Go item-by-item and write its size and contents
    for (auto &pair : cache) {
      size_t len = pair.first.size();
      out_file.write(reinterpret_cast<const char *>(&len), sizeof(size_t));
      out_file.write(pair.first.c_str(), len);
      size_t vec_size = pair.second.embedding.size();
      out_file.write(reinterpret_cast<const char *>(&vec_size), sizeof(size_t));
      out_file.write(
          reinterpret_cast<const char *>(pair.second.embedding.data()),
          vec_size * sizeof(float));
      out_file.write(reinterpret_cast<const char *>(&pair.second.mtime),
                     sizeof(time_t));
      out_file.write(reinterpret_cast<const char *>(&pair.second.id),
                     sizeof(int));
    }

    alg_hnsw->saveIndex("/home/devansh/repos/nexus/hnsw_index.bin");
  }

  // Load all entries from the disk
  void load_from_disk() {

    // Acquire a lock
    std::unique_lock<std::shared_mutex> lock(rw_lock);

    // Attempt to load the file
    std::string db_path = "/home/devansh/repos/nexus/vector_store.bin";
    std::ifstream in_file(db_path, std::ios::in | std::ios::binary);
    if (!in_file.is_open()) {
      std::cout << "[VecFS] No previous memory found. Starting fresh.\n";
      return;
    }

    // Reset the cache
    cache.clear();

    // Read the number of entires
    size_t cache_size;
    in_file.read(reinterpret_cast<char *>(&cache_size), sizeof(size_t));

    // Read one by one from the store
    while (cache_size--) {
      size_t len;
      in_file.read(reinterpret_cast<char *>(&len), sizeof(size_t));
      std::string file_path(len, '\0');
      in_file.read(&file_path[0], len);
      size_t vec_size;
      in_file.read(reinterpret_cast<char *>(&vec_size), sizeof(size_t));
      std::vector<float> embedding(vec_size);
      in_file.read(reinterpret_cast<char *>(embedding.data()),
                   vec_size * sizeof(float));
      time_t mtime;
      in_file.read(reinterpret_cast<char *>(&mtime), sizeof(time_t));
      int id;
      in_file.read(reinterpret_cast<char *>(&id), sizeof(int));
      cache[file_path] = {embedding, mtime, id};

      path_to_id[file_path] = id;
      id_to_path[id] = file_path;

      if (id >= current_id)
        current_id = id + 1;
    }

    delete alg_hnsw;
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(
        space, "/home/devansh/repos/nexus/hnsw_index.bin");
  }
};
