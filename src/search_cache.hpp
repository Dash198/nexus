#include <cstddef>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <algorithm>
#include <numeric> // for std::inner_product (cosine similarity)

// A single cached search result
struct CacheEntry {
    std::string ui_output;               // The formatted string printed to the terminal
    std::vector<float> query_embedding;  // The 384-dimensional ONNX vector of the search
    float threshold_score;               // The Cosine Similarity score of the 5th place file
    std::vector<std::string> top_paths;  // Top k paths for the query

    // Iterator pointing to this query's position in the timeline (for instant O(1) deletion)
    std::list<std::string>::iterator lru_pointer;
};

class SearchCache {
private:
    size_t capacity; // e.g., 50 max searches

    // LRU Cache
    // Pushes newest searches to the front, pops oldest from the back
    std::list<std::string> lru_list;

    // Maps a query string (e.g., "mishima") to its actual CacheEntry data
    std::unordered_map<std::string, CacheEntry> cache_map;

    // Maps a file path (e.g., "/tekken.txt") to all queries that rely on it
    std::unordered_map<std::string, std::vector<std::string>> dependency_matrix;

    // Allows multiple terminal tabs to read at once, but locks everything during a surgical strike
    mutable std::shared_mutex rw_lock;

    // Helper math function for the Intrusion Check
    float calculate_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        float dot_product = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
        float norm_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
        float norm_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));
        if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
        return dot_product / (norm_a * norm_b);
    }

public:
    SearchCache(size_t max_capacity = 50) : capacity(max_capacity) {}

    const std::string& get(std::string query){
        std::unique_lock lock(rw_lock);

        if(cache_map.find(query)==cache_map.end()){
            static const std::string empty_str = "";
            return empty_str;
        }

        struct CacheEntry& entry = cache_map[query];
        lru_list.splice(lru_list.begin(), lru_list, entry.lru_pointer);

        return entry.ui_output;
    }

    void delete_query(const std::string& victim){
        struct CacheEntry& entry = cache_map[victim];
        for(auto &path: entry.top_paths){
            auto& vec = dependency_matrix[path];
            vec.erase(std::remove(vec.begin(), vec.end(), victim), vec.end());
        }
        lru_list.erase(entry.lru_pointer);
        cache_map.erase(victim);
    }

    void put(const std::string& query, const std::vector<float>& query_embedding, const std::string& ui_output, const std::vector<std::string>& top_5_paths, float threshold){
        std::unique_lock lock(rw_lock);

        lru_list.push_front(query);
        cache_map[query] = {ui_output, query_embedding, threshold, top_5_paths, lru_list.begin()};

        for(auto &path: top_5_paths){
            dependency_matrix[path].push_back(query);
        }

        if(lru_list.size()>capacity){
            std::string victim = lru_list.back();
            delete_query(victim);
        }
    }

    void invalidate_on_edit(const std::string& file_path, const std::vector<float>& new_embedding){
        std::unique_lock lock(rw_lock);

        std::unordered_set<std::string> queries_to_kill;

        if(dependency_matrix.find(file_path) != dependency_matrix.end()){
            std::vector<std::string>& queries = dependency_matrix[file_path];
            for(auto& query: queries){
                queries_to_kill.insert(query);
            }
        }

        for(auto& kv: cache_map){
            const std::string& query = kv.first;
            struct CacheEntry& entry = kv.second;

            float score = calculate_cosine_similarity(new_embedding, entry.query_embedding);
            if(score > entry.threshold_score){
                queries_to_kill.insert(query);
            }
        }

        for(auto& query: queries_to_kill){
            delete_query(query);
        }

    }
};
