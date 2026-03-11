#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <sstream>

// Custom WordPiece Tokenizer
class Tokenizer {
    private:
    std::unordered_map<std::string, int64_t> vocab;
  public:
  Tokenizer(){
    // Set up the file stream.
    std::string vocab_path = "./vocab.txt";
    std::ifstream inFile(vocab_path);

    if(!inFile.is_open()){
        std::cout << "[NEXIUS] Unable to Open Tokenizer Vocabulary!\n";
    }

    std::string line;
    int idx = 0;

    // Load up the map (ignoring spl tokens for now)
    while(std::getline(inFile, line)){
        vocab[line] = idx;
        idx++;
    }


  }

  // Preprocess a given string before handing it over to the tokenizer
  std::vector<std::string> preprocess(std::string input){
      // Convert string to lowercase
      std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c){return std::tolower(c);});

      // Split it on spaces.
      std::vector<std::string> parts;
      std::stringstream ss(input);
      std::string token;

      while(std::getline(ss, token, ' ')){
          parts.push_back(token);
      }

      return parts;
  }

  // Encode a given string
  std::vector<int64_t> encode(std::string s){
      std::vector<std::string> tokens = preprocess(s);
      std::vector<int64_t> int_tokens;

      int_tokens.push_back(vocab["[CLS]"]);

      for(auto &word: tokens){
          int start = 0;
          bool is_unk = false;
          std::vector<int64_t> buff;
          while(start < word.length()){
              bool match_found = false;
              for(int end = word.length(); end>start; end--){
                  std::string slice = word.substr(start, end-start);
                  if(start>0){
                      slice = "##" + slice;
                  }
                  if(vocab.find(slice) != vocab.end()){
                      buff.push_back(vocab[slice]);
                      start = end;
                      match_found = true;
                      break;
                  }
              }

              if(!match_found){
                  is_unk = true;
                  break;
              }
          }

          if(is_unk){
              int_tokens.push_back(vocab["[UNK]"]);
          }
          else{
              int_tokens.insert(int_tokens.end(), buff.begin(), buff.end());
          }
      }

      int_tokens.push_back(vocab["[SEP]"]);
      return int_tokens;
  }

};
