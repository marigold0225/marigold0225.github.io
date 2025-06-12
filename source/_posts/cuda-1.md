---
title: cuda-1
date: 2024-08-17 15:16:16
tags:
  - cuda
---

编程过程中一般都是这样一个流程：本地有data，有config文件需要先读取到CPU，然后计算逻辑写在GPU，将data和config的配置文件传给GPU，算完之后的结果再传给CPU，最后CPU再写入到本地。所以第一件事就是搞清楚数据在cpu和gpu上的传输。

<!--more-->

# 第一章 获取GPU数据

## cudaMalloc

前面提到，`__global__`函数不能有返回值，类型必须要void，那么我们怎么获得GPU的数据呢？答案是指针：

```cpp
#include <cstdio>
#include <cuda_runtime.h>
__global__ void kernel(int *pret) { *pret = 42; }

auto main() -> int
{
    int ret = 0; (栈上分配)
    kernel<<<1, 1>>>(&ret);
    //int *pret = (int *)malloc(sizeof(int)); (堆上分配)
    //kernel<<<1, 1>>>(pret);
    cudaDeviceSynchronize();
    printf("%d\n", ret);
    return 0;
}
```

这段代码无论是栈上分配内存还是堆上分配内存，传给GPU都不会改变传入的指针的值，根本原因在于这都是cpu上的内存，而cpu和GPU有着各自独立的内存，分别称为主机内存和设备内存，后者也叫显存，为什么要独立的显存，因为cpu上的内存太慢，所以GPU自己搭载了一块内存。因此无论是栈上还是堆上，都是cpu的内存，GPU自然无法访问到，需要cuda特有的内存分配函数cudaMalloc：

```cpp
#include <cstdio>
#include <cuda_runtime.h>
__global__ void kernel(int *pret) { *pret = 42; }

auto main() -> int
{
    int *pret;
    cudaMalloc(&pret, sizeof(int));
    kernel<<<1, 1>>>(pret);
    cudaDeviceSynchronize();

    int ret;
    cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", ret);
    cudaFree(pret);
    return 0;
}
❯ ./a.out
42
```

上述代码，先通过cudaMalloc在GPU上分配内存，此时kernel函数便能访问到这块地址，但是这块地址cpu是不能访问的！所以我们还需要cudaMemcpy拷贝数据到cpu。最后cudaFree释放掉GPU上的内存。
看起来好像很麻烦，拷来拷去，不就影响性能了吗？

## Unified Memory

有一种新的特性，叫统一内存(managed)，cudaMallocManaged代替cudaMalloc，这样会同时在cpu和gpu上分配地址，并且内存地址使一模一样的，假如gpu上有数据写入到这个地址，cpu上的地址会自动同步数据，这样就省略了手动调用cudaMemcpy。

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel(int *pret) { *pret = 42; }

auto main() -> int
{
    int *pret;
    cudaMallocManaged(&pret, sizeof(int));
    kernel<<<1, 1>>>(pret);
    cudaDeviceSynchronize();
    printf("result = %d\n", *pret);
    cudaFree(pret);
    return 0;
}
❯ ./a.out
result = 42
```

## 分配数组
可以用```n*sizeof(int)```分配大小为n的整形数组，这样会有n个连续的int数据排列在内存中，arr指向起始地址，将arr传入核函数，就可以通过下标访问数组元素：

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel(int *arr, int n)
{
    for (int i = 0; i < n; i++) arr[i] = i;
}

auto main() -> int
{
    int  n   = 4;
    int *arr = nullptr;
    cudaMallocManaged(&arr, n * sizeof(int));
    kernel<<<1, 1>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) printf("arr[%d]: %d\n", i, arr[i]);
    cudaFree(arr);
    return 0;
}
❯ ./a.out
arr[0]: 0
arr[1]: 1
arr[2]: 2
arr[3]: 3
```
上面的代码只启动了一个线程一个block，核函数内部的for循环是串行的，这里可以因为数组大小是n，我们可以把线程数量设置为n，将数组下标转化为线程的编号，这样就扔掉了for循环：
```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel(int *arr, int n)
{
    int i  = threadIdx.x;
    arr[i] = i;
}

auto main() -> int
{
    int  n   = 4;
    int *arr = nullptr;
    cudaMallocManaged(&arr, n * sizeof(int));
    kernel<<<1, n>>>(arr, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) printf("arr[%d]: %d\n", i, arr[i]);
    cudaFree(arr);
    return 0;
}
❯ ./a.out
arr[0]: 0
arr[1]: 1
arr[2]: 2
arr[3]: 3
```
网格
