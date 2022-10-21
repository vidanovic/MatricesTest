cmake_minimum_required(VERSION 3.23)

project(MatricesTest VERSION 1.0.0 LANGUAGES CXX)

set(target_name ${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

# Include Eigen
configure_file(CMakeListsEigen.txt.in ${CMAKE_BINARY_DIR}/eigen-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/eigen-download)
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/eigen-download)

include_directories(${CMAKE_BINARY_DIR}/eigen-src)

file(GLOB SOURCES_CXX "src/*.cxx")
file(GLOB SOURCES_HXX "src/*.hxx")
list(APPEND SOURCES ${SOURCES_HXX} ${INCLUDE_HXX} ${SOURCES_CXX})

add_executable(${target_name} ${SOURCES})