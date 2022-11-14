./build_android.sh $1

ANDROID_DIR=/data/local/tmp/simd
ANDROID_OUT_DIR=/data/local/tmp/simd-result

# adb shell rm -rf $ANDROID_DIR
adb shell mkdir -p $ANDROID_DIR

ARM_ARCH=armeabi-v7a

BUILD_DIR=build/android/${ARM_ARCH}
OPENCV_DIR=thirdparty/opencv/android/lib/${ARM_ARCH}

#push .so files
# find $OPENCV_DIR -name "*.so" | while read solib; do
#     adb push $solib  $ANDROID_DIR
#     echo $solib
# done

#push executable file
adb push $BUILD_DIR/test/run_test.out $ANDROID_DIR

adb shell "export LD_LIBRARY_PATH=$ANDROID_DIR && cd $ANDROID_DIR && ./run_test.out"

adb pull $ANDROID_OUT_DIR ./result