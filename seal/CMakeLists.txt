cmake_minimum_required(VERSION 3.12)

project(onehot VERSION 1.0.0 LANGUAGES CXX C)


#set(CMAKE_CXX_FLAGS "-g -O0 -fpermissive -fopenmp  -lgmp")
set(CMAKE_CXX_FLAGS "-g -O3 -march=native -fpermissive -Ofast -std=c++11 -fopenmp")


find_package(Threads REQUIRED)
#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(SEAL 3.6.5 EXACT REQUIRED)


add_executable(test_main_cpp test1.cpp)
#target_link_libraries(test1 Threads::Threads Eigen3::Eigen SEAL::seal)
target_link_libraries(test_main_cpp Threads::Threads SEAL::seal)
target_include_directories(test_main_cpp PRIVATE)
