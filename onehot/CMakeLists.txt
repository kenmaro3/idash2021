cmake_minimum_required(VERSION 3.12)

project(onehot VERSION 1.0.0 LANGUAGES CXX C)


#set(CMAKE_CXX_FLAGS "-g -O0 -fpermissive -fopenmp  -lgmp")
set(CMAKE_CXX_FLAGS "-g -O0 -fpermissive -std=c++11")


find_package(Threads REQUIRED)


add_executable(test1 test1.cpp)
target_link_libraries(test1 Threads::Threads)
target_include_directories(test1 PRIVATE)
