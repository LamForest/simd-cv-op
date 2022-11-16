
if [[ "$1" != "fast" ]];then
    ./build_android.sh
else
    ./build_android.sh fast
fi

ANDROID_DIR=/data/local/tmp/simd
ANDROID_OUT_DIR=/data/local/tmp/simd-result

# adb shell rm -rf $ANDROID_DIR
adb shell mkdir -p $ANDROID_DIR

ARM_ARCH=armeabi-v7a

BUILD_DIR=build/android/${ARM_ARCH}
OPENCV_DIR=thirdparty/opencv/android/lib/${ARM_ARCH}

#push .so files
if [[ "$1" != "fast" ]];then
    find $OPENCV_DIR -name "*.so" | while read solib; do
        adb push $solib  $ANDROID_DIR
        echo $solib
    done
fi


#push executable file
adb push  $ANDROID_DIR
adb push $BUILD_DIR/test/run_test.out $ANDROID_DIR
adb shell "cd $ANDROID_DIR; chmod 777 run.sh;"

# adb shell "export LD_LIBRARY_PATH=$ANDROID_DIR; cd $ANDROID_DIR; ./run_test.out > ${ANDROID_OUT_DIR}/log.txt" 
adb shell "export LD_LIBRARY_PATH=$ANDROID_DIR; cd $ANDROID_DIR; ./run_test.out" 

# adb pull $ANDROID_OUT_DIR ./result