



# Depthswise Convolution算子优化

github地址

这篇文章探讨了在armv7架构下，depthwise  3x3 convolution的优化方式，并在不同尺寸的feature map下进行测试。

测试共分为两部分，第一部分是正确性测试：将优化后的算子输出和原始版本的算子输出做比较，保证优化不影响算子的结果。第二部分是耗时测试，设计了20组不同尺寸的feature map作为算子的输入，由于typora markdown表格的列数有限制，这里仅放上具有代表性的数据。

耗时测试采用MNN的策略1）进行一定次数的warm up；2）循环一定次数后挂起一段时间后，再继续执行。

| 实现方式 / 时间(ms) / C H W      | 8, 16, 16 | 8, 64, 64  | 32, 256, 256 | 128, 256, 256 | 512, 512, 512 |
| -------------------------------- | --------- | ---------- | ------------ | ------------- | ------------- |
| [1] naive_conv **(groundtruth)** | 0.0613    | 0.465      | 31.188       | 125.076       | 1881.419      |
| [2.1] naive_3x3                  | 0.005     | 0.0799     | 5.408        | 21.656        | 346.112       |
| [2.2] naive_3x3_intrin           | 0.003     | 0.0513     | 3.512        | 14.174        | 247.541       |
| [3] 2row                         | 0.004     | 0.2248     | 4.549        | 18.991        | 316.509       |
| [4.1] 4col                       | 0.005     | 0.2618     | 5.868        | 23.463        | 394.895       |
| [4.2] 4col_intrin                | **0.001** | **0.0862** | **1.998**    | **8.797**     | **155.627**   |
| [5.1] 2row_4col_intrin           | 0.001     | 0.0816     | 2.489        | 10.925        | 192.796       |
| [5.2] 2row_4col_asm **(ncnn)**   | 0.002     | 0.1117     | 2.258        | 9.317         | 185.043       |

测试结果如上表。第一列为优化策略：

- [1] naive_conv，原始实现，作为baseline和groundtruth。代码：`op/arm/dwconv3x3.cpp @naive_dwconv`
- [2.1] naive_3x3，原始3x3实现，循环展开。 代码：`op/arm/dwconv3x3.cpp @naive_dwconv3x3`
- [2.2] 用neon intrinsics优化[2.1]。代码：`op/arm/dwconv3x3_intrinsics.cpp @naive_dwconv3x3_intrinsics`
- [3] 列方向上循环展开，同时计算2行的卷积。代码：`op/arm/dwconv3x3.cpp @dwconv3x3_2row`
- [4.1] 行方向上循环展开，同时计算4列的卷积。代码：`op/arm/dwconv3x3.cpp @dwconv3x3_4col`
- [4.2] 用neon intrinsics优化[4.1]，**最优实现**。代码：`op/arm/dwconv3x3_intrinsics.cpp @dwconv3x3_4col_intrinsics`
- [5.1] 行、列方向同时展开，并用intrinics进行优化。代码：`op/arm/dwconv3x3_intrinsics.cpp @dwconv3x3_2row4col_intrinsics`
- [5.2] 用neon assembly优化 [5.2]，为ncnn的实现，其中的注释是 [A] 中贡献的，十分感谢。代码：`op/arm/dwconv3x3_asmm.cpp @dwconv3x3_4col_intrinsics`





## [1] naive_conv

上表中，[1] 为 使用不经过任何优化的，最原始方法（5重循环）实现的、适用于任何尺寸卷积核的dwconv：

```c++
for(int ch = 0; ch < C; ++ch){
	for(int h = 0; h < H; ++h){
		for(int w = 0; w < W; ++w){
			//卷积运算
			for(int k_h = 0; k_h < KH; ++k_h){
				for(int k_w = 0; k_w < KW; ++k_w){
					out[ch][h][w] += in[ch][h+k_h][w+k_w] kernel[k_h][k_w];
				}
			}
		}
	}
}
```

该实现作为groundtruth，用于比对经过了优化的算子的输出是否一致。

同时，也作为一个baseline，与优化的算子进行比较。

## [2.1] naive_3x3

这篇文章的主题是优化3x3depthwise卷积(dwconv)。为什么把仅讨论3x3的dwconv卷积上，而不是任意尺寸的dwconv（甚至conv）呢？

因为，相比于针对所有情况做优化，只对3x3卷积优化会容易很多。其次，针对所有尺寸的conv的优化方案一定没有只针对3x3卷积的快，通用性和优化程度是不能兼得的。另外，3x3卷积是深度学习里面最最常见的算子，也是占用时间最多的算子，对它进行特殊的优化是必然的。

如果只关注3x3 dwconv，在不使用任何simd、cache、访存技巧的情况下，就可以进行大幅度的优化。

观察[1]中的最里面2层循环，对于3x3卷积，这两层循环翻译而成的机器指令中，只有9次浮点的fmla操作是有意义的，其他指令都是for循环的控制语句。既然如此，可以将这两层循环手动展开，

```c++
float k0 = kth_kernel[0], k1 = kth_kernel[1], k2 = kth_kernel[2];
float k3 = kth_kernel[3], k4 = kth_kernel[4], k5 = kth_kernel[5];
float k6 = kth_kernel[6], k7 = kth_kernel[7], k8 = kth_kernel[8];

for(int ch = 0; ch < C; ++ch){
	for(int h = 0; h < H; ++h){
		for(int w = 0; w < W; ++w){
			//卷积运算
      float a0 = in_0[w + 0], a1 = in_0[w + 1], a2 = in_0[w + 2];
      float a3 = in_1[w + 0], a4 = in_1[w + 1], a5 = in_1[w + 2];
      float a6 = in_2[w + 0], a7 = in_2[w + 1], a8 = in_2[w + 2];

      out[w] = a0 * k0 + a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5 + a6 * k6 + a7 * k7 + a8 * k8;
		}
	}
}
```



这样做至少有3点好处：

1. 省去了最里面2层循环的控制语句
2. 不需要执行for循环中的分支预测，减少了分支预测失败导致流水线清空次数。 
3. 卷积运算的9次访存和fmla放在了一起，处理器在执行指令的时候拥有更广的范围进行指令重排、乱序，每个周期可以发射多条指令。

进行了如上优化后，相比于[1]的1881ms，只需要346 ms，80%的优化幅度，非常可观。



## [2.2] naive_3x3_intrin

在[2.1] 的 基础上，将卷积运算用neon intrinsics 进行优化。

具体思路：将3x3卷积拆成3个长度为3的向量之间的逐点相乘，然后做reduce sum。

```c++
//在内存中，第k个3行3列的卷积核(kth_kernel)在内存中的排布方式为 `#0,#1,#2,#3,#4,#5,#6,#7,#8`
//分别读入卷积核的每一行
//读入kernel的 第0,1,2,3元素，并把最后一个lane置0
float32x4_t k0 = vld1q_f32(kth_kernel); //k0 = [#0, #1, #2, #3]
k0 = vsetq_lane_f32(0.0f, k0, 3); //k0 = [#0, #1, #2, 0]

float32x4_t k1 = vld1q_f32(kth_kernel + 3);
k1 = vsetq_lane_f32(0.0f, k1, 3); //k1 = [#3, #4, #5, 0]

float32x4_t k2 = vld1q_f32(kth_kernel + 6);
k2 = vsetq_lane_f32(0.0f, k2, 3); //k2 = [#6, #7, #8, 0]

for(int ch = 0; ch < C; ++ch){
	for(int h = 0; h < H; ++h){
		for(int w = 0; w < W; ++w){
			//卷积运算
      //分别读入feature map 3x3位置的每一行
      float32x4_t a0 = vld1q_f32(in_0); 
      float32x4_t a1 = vld1q_f32(in_1);
      float32x4_t a2 = vld1q_f32(in_2);
			//mulltiply and add
      float32x4_t sum = vmulq_f32(a0, k0);
      sum = vmlaq_f32(sum, a1, k1);
      sum = vmlaq_f32(sum, a2, k2);
			//reduce sum
      float32x2_t s = vpadd_f32(vget_high_f32(sum), vget_low_f32(sum));
      out[w] = vget_lane_f32(s, 0) + vget_lane_f32(s, 1);
      
      in_0 += 1;
      in_1 += 1;
      in_2 += 1;
		}
	}
}
```

进行了如上优化后，相比于[2.1]的346ms，只需要247ms，提升了24%。



### [3] 2row

观察下图，在做黄色部分feature map的卷积时，从内存中加载了第0、1、2行的元素。其中，1、2行中的元素在计算蓝色部分卷积的时候也会使用到。这个时候就可以尝试做访存的复用。

![image-20221116020438163](https://raw.githubusercontent.com/LamForest/pics/main/macm1pro/202211160204191.png)

```c++
float k0 = kth_kernel[0], k1 = kth_kernel[1], k2 = kth_kernel[2];
float k3 = kth_kernel[3], k4 = kth_kernel[4], k5 = kth_kernel[5];
float k6 = kth_kernel[6], k7 = kth_kernel[7], k8 = kth_kernel[8];

for(int ch = 0; ch < C; ++ch){
	for(int h = 0; h < H; h += 2){
		for(int w = 0; w < W; ++w){
			//卷积运算
      float a0 = in_0[w + 0], a1 = in_0[w + 1], a2 = in_0[w + 2];
      float a3 = in_1[w + 0], a4 = in_1[w + 1], a5 = in_1[w + 2];
      float a6 = in_2[w + 0], a7 = in_2[w + 1], a8 = in_2[w + 2];
      float a9 = in_3[w + 0], a10 = in_3[w + 1], a11 = in_3[w + 2];

      out_0[w] =
        a0 * k0 + a1 * k1 + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5 + a6 * k6 + a7 * k7 + a8 * k8;

      out_1[w] =
        a3 * k0 + a4 * k1 + a5 * k2 + a6 * k3 + a7 * k4 + a8 * k5 + a9 * k6 + a10 * k7 + a11 * k8;
		}
	}
}
```

这样的优化策略虽然没有降低计算量，但是显著减少了访存次数。相比[2.1]有 10% 左右的提升。



## [4.1] [4.2] 4col 4col_intrin

[3]中每一次循环会同时计算相邻两行的卷积，另一种类似的思路是将卷积在行方向上循环展开，同时计算4列的卷积。

![image-20221116104403105](https://raw.githubusercontent.com/LamForest/pics/main/macm1pro/202211161044154.png)

```c++
float k0 = kth_kernel[0], k1 = kth_kernel[1], k2 = kth_kernel[2];
float k3 = kth_kernel[3], k4 = kth_kernel[4], k5 = kth_kernel[5];
float k6 = kth_kernel[6], k7 = kth_kernel[7], k8 = kth_kernel[8];

for(int ch = 0; ch < C; ++ch){
	for(int h = 0; h < H; ++h){
		for(int w = 0; w < W; w += 4){                    
			float s0, s1, s2, s3;
      float a0 = in_0[w + 0], a1 = in_0[w + 1], a2 = in_0[w + 2], a3 = in_0[w + 3], a4 = in_0[w + 4], a5 = in_0[w + 5];
			//1）
      s0 = a0 * k0 + a1 * k1 + a2 * k2;
      s1 = a1 * k0 + a2 * k1 + a3 * k2;
      s2 = a2 * k0 + a3 * k1 + a4 * k2;
      s3 = a3 * k0 + a4 * k1 + a5 * k2;
			
      a0 = in_1[w + 0], a1 = in_1[w + 1], a2 = in_1[w + 2], a3 = in_1[w + 3], a4 = in_1[w + 4], a5 = in_1[w + 5];
			//2
      s0 += a0 * k3 + a1 * k4 + a2 * k5;
      s1 += a1 * k3 + a2 * k4 + a3 * k5;
      s2 += a2 * k3 + a3 * k4 + a4 * k5;
      s3 += a3 * k3 + a4 * k4 + a5 * k5;

      a0 = in_2[w + 0], a1 = in_2[w + 1], a2 = in_2[w + 2], a3 = in_2[w + 3], a4 = in_2[w + 4], a5 = in_2[w + 5];
			//3
      s0 += a0 * k6 + a1 * k7 + a2 * k8;
      s1 += a1 * k6 + a2 * k7 + a3 * k8;
      s2 += a2 * k6 + a3 * k7 + a4 * k8;
      s3 += a3 * k6 + a4 * k7 + a5 * k8;

      cur_out_line[w] = s0;
      cur_out_line[w + 1] = s1;
      cur_out_line[w + 2] = s2;
      cur_out_line[w + 3] = s3;
    }
	}
}
```

有些意外的是，这种写法让结果变差了，512x512x512情况下，[2.1] naive_3x3只需要346ms，然而这种方法却需要 394ms，反而变慢了。

虽然如此，但是观察上面的代码，却很容易发现可用SIMD进行并行的地方。比如上面的代码的 1）2）3）处非常规整，可以很轻易的转换为向量化的表示:
$$
\mathbf{s} = k_0\mathbf{a_{0-3}} + k_1\mathbf{a_{1-4}} + k_2\mathbf{a_{2-5}} \\
\mathbf{s} += k_3\mathbf{a_{0-3}} + k_4\mathbf{a_{1-4}} + k_5\mathbf{a_{2-5}} \\
\mathbf{s} += k_6\mathbf{a_{0-3}} + k_7\mathbf{a_{1-4}} + k_8\mathbf{a_{2-5}}
$$

Neon Intrinsics实现如下：

```c++
                    /* row 0 */
                    float32x4_t r_left = vld1q_f32(in_0);
                    float32x4_t r_right = vld1q_f32(in_0 + 4);

                    float32x4_t r1 = vextq_f32(r_left, r_right, 1);
                    float32x4_t r2 = vextq_f32(r_left, r_right, 2);

                    float32x4_t s0 = vmulq_n_f32(r_left, k0);
                    s0 = vmlaq_n_f32(s0, r1, k1);
                    s0 = vmlaq_n_f32(s0, r2, k2);

                    /* row 1 */

                    r_left = vld1q_f32(in_1);
                    r_right = vld1q_f32(in_1 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s0 = vmlaq_n_f32(s0, r_left, k3);
                    s0 = vmlaq_n_f32(s0, r1, k4);
                    s0 = vmlaq_n_f32(s0, r2, k5);

                    /* row 2 */

                    r_left = vld1q_f32(in_2);
                    r_right = vld1q_f32(in_2 + 4);

                    r1 = vextq_f32(r_left, r_right, 1);
                    r2 = vextq_f32(r_left, r_right, 2);

                    s0 = vmlaq_n_f32(s0, r_left, k6);
                    s0 = vmlaq_n_f32(s0, r1, k7);
                    s0 = vmlaq_n_f32(s0, r2, k8);

                    vst1q_f32(cur_out_line + w, s0);

                    in_0 += 4;
                    in_1 += 4;
                    in_2 += 4;
```

其中用到了一个特别的指令：[vextq_f32](https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/coding-for-neon---part-5-rearranging-vectors)。假设从内存中连续加载了 8 个 float32，分别存在两个128位寄存器 q1, q2中。此时 `q1 = [a,b,c,d]; q2 = [e,f,g,h]`，如果我想要获得一个 `q3 = [d,e,f,g]`，会比较麻烦，需要先移位，再做get_lane和set_lane，不考虑互锁的情况下，至少需要3个周期。`vextq_f32`就可以完美的解决这个问题。

![ VEXT extracing new vector of bytes](https://community.arm.com/resized-image/__size/1040x0/__key/communityserver-blogs-components-weblogfiles/00-00-00-21-42/0045.blogentry_2D00_0_2D00_091652700_2B00_1331552038_5F00_thumb.png)

用Neon intrinsics优化过后，512x512x512情况下，仅需155ms，是本文尝试的优化方案中是最快的。在其余测试的feature map尺寸下，也都是如此。

### ## [5.1] [5.2] 2row + 4col

能不能将[3]中的列方向循环展开 和 [4]中的行方向循环展开结合起来呢？在这里我也尝试了一下这种策略，并用neon intrinsics做优化，即表格中的[5.1] 2row_4col_intrin这一行。

然而结果并不尽如人意，没有产生 1 + 1 >= 2的效果，相比[4.2]4col_intrin要慢上一些。

事实上，根据 [A] 和 [B] ，ncnn中采用就是这种策略，只不过为了更极致的性能，直接用neon assembly写内联汇编。不过，就算是汇编，也比instrinsics实现的[4.2]4col_intrin要慢一点。





## 参考资料

[A] [基于NCNN的3x3可分离卷积再思考盒子滤波](http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20%E7%A7%BB%E5%8A%A8%E7%AB%AF%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96/%E5%9F%BA%E4%BA%8ENCNN%E7%9A%843x3%E5%8F%AF%E5%88%86%E7%A6%BB%E5%8D%B7%E7%A7%AF%E5%86%8D%E6%80%9D%E8%80%83%E7%9B%92%E5%AD%90%E6%BB%A4%E6%B3%A2/)

[B] [ncnn depthwise 实现](https://github.com/Tencent/ncnn/blob/master/src/layer/arm/convolutiondepthwise_3x3.h)



# 其他

dwconv可以用pack(nc4hw4)的数据排布吗？

深度可分离卷积（Depthwise Separable Convolution）由Depthwise Convolution（dwconv）和Pointwise Convolution（pwconv）组成。

其中 dwconv 可以看作 #Group = #Channel 的分组卷积，是为了降低传统卷积计算量而设计出来的特殊卷积。不过，dwconv有一些缺点，比如 1）卷积只在独立的channel中进行，没有channel之间的信息的融合，降低了模型的容量？。 2）dwconv的输出的通道数只能和输出保持一样。考虑到这两点，dwconv之后一般要接一个pwconv。

这里讨论的是dwpconv的实现及优化
