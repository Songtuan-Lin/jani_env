cmake_minimum_required(VERSION 3.10)

# Project name and language
project(jani_engine LANGUAGES CXX)

# Require at least C++17 (you can change this if needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Collect all .cpp files in src/
file(GLOB_RECURSE PROJECT_SOURCES src/*.cpp)

# Create the executable (name: jani_engine)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})