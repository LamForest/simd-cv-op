rm -rf build/mac
mkdir -p build/mac && cd build/mac
cmake -G Xcode -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=./ \
../../