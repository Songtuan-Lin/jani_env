cmake_minimum_required(VERSION 3.14)

# Project name and language
project(jani_engine LANGUAGES CXX)

# Require at least C++20 (you can change this if needed)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Collect all .cpp files in src/
file(GLOB_RECURSE PROJECT_SOURCES src/engine.cpp src/expressions.cpp src/main.cpp)

# Create the executable (name: jani_engine)
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Link GoogleTest to the project
enable_testing()

add_executable(expression_test test/unittest_expr.cpp src/expressions.cpp)
target_link_libraries(expression_test GTest::gtest_main)
target_include_directories(expression_test
    PRIVATE
        src/
)
include(GoogleTest)
gtest_discover_tests(expression_test)