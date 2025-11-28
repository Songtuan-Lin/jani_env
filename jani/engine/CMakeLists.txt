cmake_minimum_required(VERSION 3.14)

# Project name and language
project(jani_engine LANGUAGES CXX)

# Require at least C++20 (you can change this if needed)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Python interpreter and Development module
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Find Boost libraries
find_package(Boost REQUIRED)

# Find nanobind
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

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

# Create the library (name: jani_engine)
add_library(${PROJECT_NAME} src/engine.cpp src/expressions.cpp src/oracle.cpp)
target_link_libraries(${PROJECT_NAME} Boost::headers)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
)

# Create nanobind module
nanobind_add_module(backend backend.cpp)
target_link_libraries(backend PRIVATE ${PROJECT_NAME})

# Link GoogleTest to the project
enable_testing()

add_executable(expression_test test/unittest_expr.cpp src/expressions.cpp)
target_link_libraries(expression_test GTest::gtest_main Boost::headers)
target_include_directories(expression_test
    PRIVATE
        src/
)

add_executable(automaton_test test/unittest_automaton.cpp src/engine.cpp src/expressions.cpp)
target_link_libraries(automaton_test GTest::gtest_main Boost::headers)
target_include_directories(automaton_test
    PRIVATE
        src/
)

add_executable(engine_test test/unittest_engine.cpp src/engine.cpp src/expressions.cpp)
target_link_libraries(engine_test GTest::gtest_main Boost::headers)
target_include_directories(engine_test
    PRIVATE
        src/
)

add_executable(oracle_test test/unittest_oracle.cpp src/engine.cpp src/expressions.cpp src/oracle.cpp)
target_link_libraries(oracle_test GTest::gtest_main Boost::headers)
target_include_directories(oracle_test
    PRIVATE
        src/
)

include(GoogleTest)
gtest_discover_tests(expression_test)
gtest_discover_tests(automaton_test)
gtest_discover_tests(engine_test)
gtest_discover_tests(oracle_test)