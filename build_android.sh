if [ ! $ANDROID_NDK ]; then
    export ANDROID_NDK="${HOME}/Library/Android/sdk/ndk/21.4.7075529"
else
    echo "ANDROID_NDK:"$ANDROID_NDK
fi

TMP_ANDROID_ABI="armeabi-v7a with NEON"

rm -rf build/android/${TMP_ANDROID_ABI}
mkdir -p build/android/${TMP_ANDROID_ABI}
cd build/android/${TMP_ANDROID_ABI}


cmake ../../../ -DBUILD_TEST=ON\
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_STL=c++_static \
    -DANDROID_NATIVE_API_LEVEL=android-21  \
    -DANDROID_TOOLCHAIN=clang \
    -DNATIVE_LIBRARY_OUTPUT=. 
    # -H../ -B$BUILD_DIR

if [ $? != 0 ]; then
    echo -e "\n\033[31m Error: Build Failed ! \033[m\n"
    exit -1
fi

make -j8
pwd
# make install
if [ $? != 0 ]; then
    echo -e "\n\033[31m Error: Build Failed ! \033[m\n"
    exit -1
fi
