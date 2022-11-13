
cmake_minimum_required(VERSION 3.6.0)
set(CMAKE_CXX_STANDARD 11)

project(SIMD)
message(STATUS "cmake project name : ${PROJECT_NAME}")

include(cmake/platform.cmake)

message("ENABLE_NEON: ${ENABLE_NEON}")

set(CMAKE_CXX_FLAGS "-DNDEBUG -O2")

file(GLOB CV_OP_SRC ./cv/arm/*.cpp ./op/arm/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/test)
include_directories(${PROJECT_SOURCE_DIR}/cv/arm)
include_directories(${PROJECT_SOURCE_DIR}/op/arm)
include_directories(${PROJECT_SOURCE_DIR}/test/cv)

option(BUILD_TEST " build test project " OFF)

if(BUILD_TEST)
    message(STATUS "CV_OP_SRC files : ${CV_OP_SRC}")

    add_subdirectory(test)
    option(OPENCV_ENABLED " enable opencv " ON)

    if(OPENCV_ENABLED)
        if(OS_MACOS)
            if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
                set(OPENCV_ARCH macos/lib/arm64)
            else()
                set(OPENCV_ARCH macos/lib/x86_64)
            endif()

            set(OPENCV_SUFFIX 4.5.dylib)
            include_directories(${PROJECT_SOURCE_DIR}/thirdparty/opencv/macos/include)
            include_directories(${PROJECT_SOURCE_DIR}/thirdparty/opencv/macos/include/opencv4)
            include_directories(${PROJECT_SOURCE_DIR}/thirdparty/opencv/macos/include/opencv4/opencv2)
        endif()

    endif()
endif()
