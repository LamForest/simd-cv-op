# simd-op-cv
这是一个记录移动端(arm)性能优化学习过程的库，分为两个部分
1. 推理引擎：
   - 算子开发&优化：[DepthwiseConv3x3优化](https://github.com/LamForest/simd-cv-op/blob/master/note/dwconv.md)（完成）、relu、全连接层。
   - 线程池：[SimpleThreadPool](https://github.com/LamForest/cpp-blogs-code/blob/master/concurrent/threadpool/pool2/pool.hpp) 用于每一层算子的数据分块
   - 模型转换及加载：todo
   - 数据排布转换(e.g. NC4HW4)：todo
  
  小目标：能够推理简单的[DepthwiseConv MNIST](https://github.com/LamForest/simd-cv-op/blob/master/pytorch_example/mnist_dwconv/main.py)模型，耗时至少与ncnn持平。

2. 图像算子：
   - [bgr2gray优化](https://github.com/LamForest/simd-cv-op/blob/master/note/bgr2gray.md)（完成）
   - boxFilter（完成）
   - integral（完成）
   - max_filter (进行中)

## 如何交叉编译并运行

### 必要：
1. 安装ndk 21.4.7075529，并放在`${HOME}/Library/Android/sdk/ndk/21.4.7075529`中；或按照自己的喜好修改`build_android.sh`
2. opencv: thirdparty中已有预编译好的mac m1 下的opencv库，如果是其他平台，需要自己编译，并放在thirdpary中。
3. 一台arm芯片安卓手机

### 编译
`./build_android.sh`

### 编译 & adb push & 运行
`./run_android_test.sh`




