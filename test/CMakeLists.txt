
file(GLOB_RECURSE Files ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/cv/*.cpp)


include_directories(${CMAKE_CURRENT_LIST_DIR}/cv)
include_directories(${CMAKE_CURRENT_LIST_DIR})



add_executable(run_test.out ${Files} ${CV_OP_SRC} )

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

  if(OS_MACOS OR OS_ANDROID)
    list(APPEND OPENCV_LIBS
      ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${OPENCV_ARCH}/libopencv_core.${OPENCV_SUFFIX}
      ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${OPENCV_ARCH}/libopencv_highgui.${OPENCV_SUFFIX}
      ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${OPENCV_ARCH}/libopencv_imgcodecs.${OPENCV_SUFFIX}
      ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${OPENCV_ARCH}/libopencv_imgproc.${OPENCV_SUFFIX}
      ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${OPENCV_ARCH}/libopencv_photo.${OPENCV_SUFFIX}
      ${PROJECT_SOURCE_DIR}/thirdparty/opencv/${OPENCV_ARCH}/libopencv_video.${OPENCV_SUFFIX}
    )
  endif()

  target_link_libraries(run_test.out ${OPENCV_LIBS})
endif()

target_include_directories(run_test.out PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)

