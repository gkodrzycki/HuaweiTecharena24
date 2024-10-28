EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

if(${ARCHITECTURE} STREQUAL "aarch64")

    # debug flags: -fsanitize=address,undefined -D_GLIBCXX_DEBUG
    set(CMAKE_C_COMPILER /usr/bin/gcc)
    set(CMAKE_CXX_COMPILER /usr/bin/g++)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,/usr/lib64/")
    link_directories("/usr/lib64/")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -lz -gdwarf-2")
    #set(HDF5_LIB_PATH  ${/home/dedup/anaconda3/lib/libhdf5.a})
    #include_directories(/home/dedup/anaconda3/lib/hdf5/include)
else()
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lhdf5")
endif()

cmake_minimum_required(VERSION 3.12.0)
project(ann LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -falign-jumps=64 -fopenmp -fPIC -Ofast -march=native -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")



include_directories(${PROJECT_SOURCE_DIR}/..)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/../third_party/helpa)

add_library(ann STATIC
        ann.cpp
        ${PROJECT_SOURCE_DIR}/ann.cpp)

add_executable(run main.cpp)


if(${ARCHITECTURE} STREQUAL "aarch64")
    target_link_libraries(run ann ${HDF5_LIB_PATH})
else()
    target_link_libraries(run ann ${HDF5_LIB_PATH} dl z)
endif()