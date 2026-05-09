#include "src/db/vector_store.hpp"
#include <dirent.h>
#include <sys/stat.h>
#include <string>
#include <cstring>
#include <iostream>
#include <unordered_set>

void explore(std::string path){
    DIR *dir = opendir(path.c_str());

    if(dir==nullptr)    return;

    struct dirent *entry;
    struct stat buf;
    while((entry=readdir(dir))!=NULL){

        if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)    continue;

        if(entry->d_type == DT_REG){
            std::string full_path = path + "/" + entry->d_name;
            lstat(full_path.c_str(), &buf);
            std::cout << "name: " << full_path << " | mtime: " << buf.st_mtime << std::endl;
        }
        else if(entry->d_type == DT_DIR){
            explore(path+"/"+entry->d_name);
        }
    }

    closedir(dir);
}

// 1. Pass the set by reference from main() so it survives recursion!
void sync_drive(std::string path, VectorStore &store, std::unordered_set<std::string>& found_files){
    DIR *dir = opendir(path.c_str());
    if(dir==nullptr) return;

    struct dirent *entry;
    struct stat buf;
    while((entry=readdir(dir))!=NULL){
        if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;

        if(entry->d_type == DT_REG){
            std::string full_path = path + "/" + entry->d_name;
            lstat(full_path.c_str(), &buf);

            // 2. Strip the base directory to match the FUSE ledger!
            std::string base_dir = "/home/devansh/repos/nexus/nexus_data";
            std::string fuse_path = full_path.substr(base_dir.length());

            // 3. Insert the FUSE path, not the absolute path
            found_files.insert(fuse_path);

            // 4. Query the store using the FUSE path
            if(!store.contains(fuse_path)){
                std::cout << "Embedding Newborn: " << fuse_path << std::endl;
            }
            else{
                if(buf.st_mtime > store.get_mtime(fuse_path)){
                    std::cout << "Mutant found: " << fuse_path << " | Actual mtime: " << buf.st_mtime << std::endl;
                }
            }
        }
        else if(entry->d_type == DT_DIR){
            sync_drive(path+"/"+entry->d_name, store, found_files);
        }
    }
    closedir(dir);
}int main(){
    VectorStore store;
    store.load_from_disk();
    std::unordered_set<std::string> s;
    sync_drive("/home/devansh/repos/nexus/nexus_data", store, s);
}
