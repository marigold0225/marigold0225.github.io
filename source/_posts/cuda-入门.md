---
title: cuda 入门
date: 2024-08-16 19:57:12
tags:
  - cuda
---

学习cuda的一些笔记。

<!--more-->

# 第0章

cuda有两套API，cuda runtime API和cuda driver API，前者简单兼顾性能，无需手动编译kernel，但不够灵活，后者灵活多变，但是操作繁琐，适合有特殊需求的用户，这个笔记只记录比较易懂，使用最多的runtime API。
先阐述一些概念和语法
## global host device
```cpp
#include <cuda_runtime.h>
#include <cstdio>

__device__ void say_hello_device() { printf("hello world from GPU!\n"); }

__host__ void say_hello_host() { printf("hello world from CPU!\n"); }

__host__ __device__ void say_hello() { printf("hello world!\n"); }


__host__ __device__ void say_hello_select() {

#ifdef __CUDA_ARCH__
    printf("hello world from GPU!\n");
#else
    printf("hello world from CPU!\n");
#endif
}


__global__ void cuda_hello() {
  say_hello_device();
  say_hello();
}

auto main() -> int
{
    cuda_hello<<<1, 10>>>();
    cudaDeviceSynchronize();
    say_hello_host();
    say_hello();
    return 0;
}
```

- `__global__`修饰的函数叫核函数，是可以在main函数中调用的，运行在GPU(device)上，需要指定特殊的参数<<<a,b>>>。
- `__device__`修饰的函数叫设备函数，只能在GPU上运行，main函数内无法调用,只能由device函数或者global函数调用。
- `__host__`修饰的函数是host函数，定义在cpu上。cuda完全兼容c++，任何函数默认都是host。
- host和device可以一起用，相当于同时把函数定义在cpu和gpu上。也可以使用`__CUDA_ARCH__`宏生成两份源代码。
- `__global__`函数可以有参数，但不能有返回值，只能是void类型。也就是说，我们不可能从核函数通过返回值来获取GPU数据。
- `__global__`是异步的，所以每个global函数执行完后必须加cudaDeviceSynchronize等待GPU计算完毕

## __CUDA_ARCH__
如果打印`__CUDA_ARCH__`，会发现这是一个整数，代表的是版本号:
```cpp
__host__ __device__ void say_hello()
{
#ifdef __CUDA_ARCH__
printf("Hello, world GPU arch %d!\n", __CUDA_ARCH__);
#else
    printf("Hello, world CPU!\n");
#endif
}
Hello, world GPU arch 520!

```
这里的520代表版本号是5.2.0，默认是兼容最老的版本52，也就是GTX900以上，有了这个就可以根据不同的GPU架构来编译出不同的源码：
```cpp
__host__ __device__ func()
{
#ifdef __CUDA_ARCH__ >= 700
    // device code path for compute capability 7.x
#elif __CUDA_ARCH__ >=600
    // device code path for compute capability 6.x
// others
#elif !defined(__CUDA_ARCH__)
    //host code
#endif
}
```
在cmake中，可以指定native来检测本地机器的计算能力，也可以自动设定为特定的版本：
```cmake
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)
```
## constexpr

相当于把constexpr 函数自动变成cpu和gpu都可以调用，通常都是一些可以内联的函数，数学表达式，不依赖cpu和gpu，没有外部副作用:

```cpp
#include <cstdio>
#include <cuda_runtime.h>

constexpr auto cuthead(const char *p) -> const char * { return p + 1; }

__global__ void kernel() { printf(cuthead("Hello, world!\n")); }

auto main() -> int
{
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf(cuthead("Hello, world!\n"));
    return 0;
}
```

编译需要添加参数:
```bash
nvcc -ccbin gcc-13 constexpr.cu --expt-relaxed-constexpr
./a.out
ello, world!
ello, world!
```

constexpr没法调用printf，也不能用__syncthreads之类的GPU函数，所以也不能完全代替`__host__`和`__device__`。

## thread block grid
```cpp
#include <cstdio>
#include <cuda_runtime.h>
__global__ void kernel()
{
    printf("Hello, world!\n"));

    printf("threadIdx.x %d of %d\n", threadIdx.x, blockDim.x);
}
auto main() -> int
{
    kernel<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
# 输出
hello world from GPU!
hello world from GPU!
hello world from GPU!
threadIdx.x 0 of 3
threadIdx.x 1 of 3
threadIdx.x 2 of 3
```
运行这个代码会打印三次，这个很奇怪的语法<<<1,3>>>是核函数独有的参数，第二个参数代表了线程的数量。GPU是为了并行而生的，可以启动相当大的线程数量，不像cpu有核数的限制。第二个打印语句中的threadIdx.x，就是CUDA的特殊变量，只有在核函数中才可以访问，可以看到打印出了0，1，2三个数字，因为我们指定了线程数为3，代表了线程的编号，第二个blockDIm则代表了线程的数量，也就是尖括号内指定的3，为什么有个.x后缀呢？如果我们把尖括号的第一个参数改成2呢？
```cpp
#include <cstdio>
#include <cuda_runtime.h>
__global__ void kernel()
{
    printf("threadIdx.x %d of %d\n", threadIdx.x, blockDim.x);
}

auto main() -> int
{
    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
threadIdx.x 0 of 3
threadIdx.x 1 of 3
threadIdx.x 2 of 3
threadIdx.x 0 of 3
threadIdx.x 1 of 3
threadIdx.x 2 of 3
```
这里就是CUDA的另一个概念，block，这是一个比线程更大的概念，block可以组织线程，一个block可以由多个线程组成，blockDim就代表一个block的维度大小，而block的数量可以用尖括号第一个参数指定。<<<2,3>>>代表我们启动了两个block，每个block启动3个线程，一共6个线程。同样，要获取block的编号类似线程编号，使用blockIdx.x，获取block的数量，则使用gridDim.x。那么这个grid又是什么呢？
```cpp
__global__ void cuda_hello()
{
    printf("blockIdx.x %d of %d, threadIdx.x %d of %d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

auto main() -> int
{
    cuda_hello<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
blockIdx.x 1 of 2, threadIdx.x 0 of 3
blockIdx.x 1 of 2, threadIdx.x 1 of 3
blockIdx.x 1 of 2, threadIdx.x 2 of 3
blockIdx.x 0 of 2, threadIdx.x 0 of 3
blockIdx.x 0 of 2, threadIdx.x 1 of 3
blockIdx.x 0 of 2, threadIdx.x 2 of 3
```
从这里就可以看出，CUDA中由block组织线程，由grid组织block，这里一共6个线程。总结一下：
- thread：最小并行单位
- block：组织若干线程
- grid：组织整个任务，包含若干block
- 调用语法：<<<gridDim,blockDim>>>

一个有用的技巧，扁平化处理这些线程和block：
总线程数量：blockDim x gridDim
总线程编号：blockDim x blockIdx + threadIdx(其实就是二维数组的一维索引)
```cpp
__global__ void cuda_hello()
{
    unsigned int tnum = gridDim.x * blockDim.x;
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("tid %d of %d\n", tid, tnum);
}

auto main() -> int
{
    cuda_hello<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
tid 3 of 6
tid 4 of 6
tid 5 of 6
tid 0 of 6
tid 1 of 6
tid 2 of 6
```
可以看到，我们设置了6个线程，并且每个线程的id都正确的计算了，下面的示意图展示了其中的关系：
![](grid.png)
### 3D grid and block
根据上面block和thread的后缀.x可以猜测，肯定有.y和.z的后缀吧。cuda支持三维的block和线程区间，只需要将尖括号内改成dim3类型即可:
```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel()
{
    printf("Block (%d, %d, %d) of (%d, %d, %d), Thread (%d, %d, %d) of (%d, %d, "
           "%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z, threadIdx.x,
           threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
}

auto main() -> int
{
    kernel<<<dim3(2, 1, 3), dim3(2, 2, 2)>>>();
    cudaDeviceSynchronize();
    return 0;
}
```
```bash
Block (0, 0, 1) of (2, 1, 3), Thread (0, 0, 0) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (1, 0, 0) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (0, 1, 0) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (1, 1, 0) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (0, 0, 1) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (1, 0, 1) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (0, 1, 1) of (2, 2, 2)
Block (0, 0, 1) of (2, 1, 3), Thread (1, 1, 1) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (0, 0, 0) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (1, 0, 0) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (0, 1, 0) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (1, 1, 0) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (0, 0, 1) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (1, 0, 1) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (0, 1, 1) of (2, 2, 2)
Block (1, 0, 0) of (2, 1, 3), Thread (1, 1, 1) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (0, 0, 0) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (1, 0, 0) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (0, 1, 0) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (1, 1, 0) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (0, 0, 1) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (1, 0, 1) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (0, 1, 1) of (2, 2, 2)
Block (1, 0, 1) of (2, 1, 3), Thread (1, 1, 1) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (0, 0, 0) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (1, 0, 0) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (0, 1, 0) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (1, 1, 0) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (0, 0, 1) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (1, 0, 1) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (0, 1, 1) of (2, 2, 2)
Block (1, 0, 2) of (2, 1, 3), Thread (1, 1, 1) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (0, 0, 0) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (1, 0, 0) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (0, 1, 0) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (1, 1, 0) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (0, 0, 1) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (1, 0, 1) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (0, 1, 1) of (2, 2, 2)
Block (0, 0, 0) of (2, 1, 3), Thread (1, 1, 1) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (0, 0, 0) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (1, 0, 0) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (0, 1, 0) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (1, 1, 0) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (0, 0, 1) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (1, 0, 1) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (0, 1, 1) of (2, 2, 2)
Block (0, 0, 2) of (2, 1, 3), Thread (1, 1, 1) of (2, 2, 2)
```
因为GPU的业务常常涉及到图形学和二维图像，所以有3维结构处理会比较方便。也可以将所有线程的id展开为一维索引，本质没有区别。
初看可能很疑惑，我只需要6个线程，直接线程数设置为6就行了，为什么要分block和grid呢？还搞成3d这样复杂的结构？其实这正是cuda编程的核心，同样的代码，如果线程的结构组织不一样，运行效率天差地别。
## 总结
介绍一些cuda编程的基本概念与语法，主机端和设备端的函数区别，了解线程的组织关系。
```
虽然cuda基于C++，但是runtime的api依然是仿c风格的接口，可能是为了照顾从c转过来的土木老哥，也有可能是为了方便被第三方二次封装
```
