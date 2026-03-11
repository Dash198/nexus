#include <iostream>
#include <vector>
#include <string>
#include "tokenizer.hpp" // Pull in your engine!

int main() {
    std::cout << "[TEST] Booting NexusTokenizer..." << std::endl;
    Tokenizer tokenizer;

    // Test cases covering standard words, sub-words, and failures
    std::vector<std::string> test_queries = {
        "hello",
        "reinforcement learning",
        "origami",
        "asdfghjklxyz" // This should trigger the [UNK] trap
    };

    for (const auto& query : test_queries) {
        std::cout << "Query: \"" << query << "\"\nTokens: [ ";

        std::vector<int64_t> tokens = tokenizer.encode(query);

        for (int64_t t : tokens) {
            std::cout << t << " ";
        }
        std::cout << "]\n------------------------\n";
    }

    return 0;
}
