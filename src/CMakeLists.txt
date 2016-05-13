cmake_minimum_required(VERSION 2.8)

project(Beamformer C CXX)

# CMAKE compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D_REENTRANT -fPIC -O3 -fopenmp -g -lm")

# Source files
set(SOURCE_FILES beamformer.cpp beamformer.hpp)

# Create shared library
add_library(beamformer SHARED ${SOURCE_FILES} )

# Create executable
add_executable(beamformer_test beamformer.cpp)
target_link_libraries(beamformer_test beamformer)

# Install library
install(TARGETS "beamformer" DESTINATION /usr/local/lib)