#pragma once
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <queue>
#include <fstream>
#include <ctime>

typedef struct {
    std::vector<float> embedding;
    time_t mtime;
} embedding_data;

class VectorStore {
private:
    // The RAM Cache: Maps an absolute filepath to its 384-dimensional vector
    std::unordered_map<std::string, embedding_data> cache;

    // Private math helper
    float cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b){
        int n = vec_a.size();

        float dp = 0.0f;
        float n_a = 0.0f, n_b = 0.0f;

        for(int i=0; i<n; i++){
            dp += vec_a[i]*vec_b[i];
            n_a += vec_a[i] * vec_a[i];
            n_b += vec_b[i] * vec_b[i];
        }

        if(n_a < 1e-8 || n_b < 1e-8){
            return 0.0f;
        }

        return (dp/(std::sqrt(n_a)*std::sqrt(n_b)));
    }

public:
    VectorStore() = default;
    ~VectorStore() = default;

    // Core CRUD operations for the FUSE hooks later
    void upsert(const std::string& filepath, const std::vector<float>& embedding){
        time_t now = time(NULL);
        cache[filepath] = {embedding, now};
    }
    void remove(const std::string& filepath){
        cache.erase(filepath);
    }

    // Helper function to check if path exists in store
    bool contains(const std::string path){
        if(cache.find(path) == cache.end())
            return false;
        return true;
    }

    // Assuming path exists, get last modified time
    time_t get_mtime(const std::string path){
        return cache[path].mtime;
    }

    // The actual search engine
    std::vector<std::string> search(const std::vector<float>& query_embedding, int top_k = 5){
        std::priority_queue<std::pair<float, std::string>> pq;

        for(auto &pair: cache){
            float score = cosine_similarity(query_embedding, pair.second.embedding);
            pq.push({score, pair.first});
        }

        std::vector<std::string> results;
        for(int i=0; i<top_k && !pq.empty(); i++){
            results.push_back(pq.top().second);
            pq.pop();
        }
        return results;
    }

    void save_to_disk(){
        std::string db_path = "/home/devansh/repos/nexus/vector_store.bin";
        std::ofstream out_file(db_path, std::ios::out | std::ios::binary);        if(!out_file.is_open()){
            std::cerr << "[NEXUS] Error opening file for writing\n";
        }

        size_t cache_size = cache.size();
        out_file.write(reinterpret_cast<const char*>(&cache_size), sizeof(size_t));

        for(auto &pair:cache){
            size_t len = pair.first.size();
            out_file.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
            out_file.write(pair.first.c_str(), len);
            size_t vec_size = pair.second.embedding.size();
            out_file.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
            out_file.write(reinterpret_cast<const char*>(pair.second.embedding.data()), vec_size * sizeof(float));
            out_file.write(reinterpret_cast<const char*>(&pair.second.mtime), sizeof(time_t));
        }

    }

    void load_from_disk(){
        std::string db_path = "/home/devansh/repos/nexus/vector_store.bin";
        std::ifstream in_file(db_path, std::ios::in | std::ios::binary);
        if(!in_file.is_open()){
            std::cout << "[NEXUS] No previous memory found. Starting fresh.\n";
            return; // <--- This saves the daemon!
        }

        cache.clear();

        size_t cache_size;
        in_file.read(reinterpret_cast<char*>(&cache_size), sizeof(size_t));

        while(cache_size--){
            size_t len;
            in_file.read(reinterpret_cast<char*>(&len), sizeof(size_t));
            std::string file_path(len, '\0');
            in_file.read(&file_path[0], len);
            size_t vec_size;
            in_file.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
            std::vector<float> embedding(vec_size);
            in_file.read(reinterpret_cast<char*>(embedding.data()), vec_size*sizeof(float));
            time_t mtime;
            in_file.read(reinterpret_cast<char*>(&mtime), sizeof(time_t));
            cache[file_path] = {embedding, mtime};
        }
    }
};
