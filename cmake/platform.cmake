if(WIN32)
    set(TARGET_PLATFORM WINDOWS)
    set(OS_WINDOWS TRUE)
    set(OS_DESKTOP TRUE)
elseif(APPLE)
    if (IOS)
        set(TARGET_PLATFORM IOS)
        set(OS_IOS TRUE)
    elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(TARGET_PLATFORM MACOS)
        set(OS_MACOS TRUE)
        set(OS_DESKTOP TRUE)
    else()
        set(TARGET_PLATFORM IPHONE_SIMULATOR)
        set(OS_IPHONE_SIMULATOR TRUE)
    endif()
    set(OS_APPLE TRUE)
elseif(UNIX)
    if (NOT ANDROID)
        set(TARGET_PLATFORM LINUX)
        set(OS_LINUX TRUE)
        set(OS_DESKTOP TRUE)
    elseif(ANDROID)
        set(TARGET_PLATFORM ANDROID)
        set(OS_ANDROID TRUE)
    endif()
else()
    message(FATAL_ERROR " unknown platform ")
endif()

if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    set(OS_64BIT TRUE)
else()
    set(OS_64BIT FALSE)
endif()

if(ANDROID)
    if(ANDROID_ABI MATCHES "^arm.*$")
        option(ENABLE_NEON "enable neon" on)
    else()
        option(ENABLE_NEON "enable neon" off)
    endif()
endif()
if(IOS)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        option(ENABLE_NEON "enable neon" off)
    else()
        option(ENABLE_NEON "enable neon" on)
    endif()
endif()
if(OS_MACOS)
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        option(ENABLE_NEON "enable neon" on)
    else()
        option(ENABLE_NEON "enable neon" off)
    endif()
endif()
