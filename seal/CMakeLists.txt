cmake_minimum_required(VERSION 3.12)

project(capsuleflow VERSION 1.0.0 LANGUAGES CXX C)


#set(CMAKE_CXX_FLAGS "-g -O0 -fpermissive -fopenmp  -lgmp")
set(CMAKE_CXX_FLAGS "-g -O0 -fpermissive -fopenmp ")


find_package(Threads REQUIRED)

# Import Microsoft SEAL
find_package(SEAL 3.6.5 EXACT REQUIRED)

add_executable(test1 test1.cpp)
target_link_libraries(test1 SEAL::seal Threads::Threads)
target_include_directories(test1 PRIVATE)
