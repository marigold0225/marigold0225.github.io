---
title: scoop
date: 2024-08-06 16:09:02
tags:
  - windows
  - wsl2
---

Windows 下的一些配置，包括powershell，scoop，wsl2等。

<!--more-->

# windows

## 安装powershell7

[选择win-x64-msi安装](https://github.com/PowerShell/PowerShell/releases)
or

```bash
winget search Microsoft.PowerShell
winget install --id Microsoft.PowerShell --source winget
```

## install scoop

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

### 关于scoop

scoop是windows上的一个强大的包管理，可惜默认的软件仓库(main bucket)数量有限，因此我们需要添加额外的软件仓库：

```bash
# list support bucket
❯ scoop bucket known
main
extras
versions
nirsoft
sysinternals
php
nerd-fonts
nonportable
java
games
# list bucket local
❯ scoop bucket list

Name   Source                                    Updated           Manifests
----   ------                                    -------           ---------
main   https://github.com/ScoopInstaller/Main    2024/8/6 12:29:26      1337
extras https://gitee.com/scoop-bucket/extras.git 2024/8/6 8:35:51       2053
```

一般来说只需要一个main和extras就够了

```bash
scoop bucket add extras
```

在scoop安装的过程中可能需要配置scoop的安装路径，但是我建议是都安装c盘，因为安装的基本都是常用而且比较小的，如编译器，rust构建的系统工具等等

### scoop 常用命令

```bash
scoop update * #更新所有包
scoop config proxy 127.0.0.1:7890 #设置代理
scoop uninstall appname
scoop search appname
scoop install appname
scoop reset appname/* #恢复app或者所有app
```

下面是一些常用软件，包括编译器，rust小工具以及一些系统级工具(无需mingw和msys2，直接就能安装gcc和llvm，不花里胡哨)

```bash
scoop list
Installed apps:

Name          Version                  Source Updated             Info
----          -------                  ------ -------             ----
7zip          24.07                    main   2024-06-20 21:02:13
bat           0.24.0                   main   2023-11-29 20:48:29
bottom        0.10.2                   main   2024-08-06 15:21:29
broot         1.41.1                   main   2024-08-05 10:24:13
cacert        2024-07-02               main   2024-07-03 10:43:22
cmake         3.30.2                   main   2024-08-04 12:29:32
curl          8.9.1_1                  main   2024-08-04 12:29:37
dark          3.14                     main   2024-07-19 13:40:25
delta         0.17.0                   main   2024-03-18 14:19:35
duf           0.8.1                    main   2023-12-09 16:24:31
dust          1.1.1                    main   2024-07-18 11:34:03
eza           0.18.24                  main   2024-08-04 12:29:42
fastfetch     2.21.0                   main   2024-08-05 16:54:07
fd            10.1.0                   main   2024-05-10 10:55:01
file          5.45                     main   2023-12-28 11:53:11
fzf           0.54.3                   main   2024-08-04 12:29:48
gcc           13.2.0                   main   2024-08-05 22:11:32
gdu           5.29.0                   main   2024-07-16 15:09:31
git           2.46.0                   main   2024-07-30 17:48:02
glow          1.5.1                    main   2024-06-22 13:37:17
go            1.22.5                   main   2024-07-03 10:44:28
grex          1.4.5                    main   2024-03-09 13:16:11
gsudo         2.5.1                    main   2024-05-31 15:47:23
helix         24.07                    main   2024-07-15 11:32:41
hyperfine     1.18.0                   main   2023-12-09 17:08:18
imagemagick   7.1.1-36                 main   2024-08-02 10:41:01
innounp       0.50                     main   2024-06-16 18:35:29
jq            1.7.1                    main   2023-12-27 16:15:36
keyviz        1.0.6                    extras 2024-06-16 18:35:35
lazygit       0.43.1                   extras 2024-07-14 12:15:12
less          661                      main   2024-07-01 20:00:40
llvm          18.1.8                   main   2024-08-05 22:12:57
lua           5.4.7-2                  main   2024-07-27 11:33:03
luarocks      3.11.1                   main   2024-07-16 01:04:19
make          4.4.1                    main   2023-11-29 21:01:44
miller        6.12.0                   main   2024-06-22 13:41:01
miniconda3    24.5.0-0                 extras 2024-07-19 13:49:32
mpv           0.38.0                   extras 2024-06-19 21:20:39
neovide       0.13.3                   extras 2024-07-18 11:34:07
neovim        0.10.1                   main   2024-07-27 12:15:40
ninja         1.12.1                   main   2024-05-18 13:33:50
nodejs        22.5.1                   main   2024-07-21 17:02:34
onefetch      2.21.0                   extras 2024-06-27 16:02:02
ouch          0.5.1                    main   2024-06-22 13:43:50
poppler       24.07.0-0                main   2024-07-27 22:41:12
procs         0.14.6                   main   2024-07-30 17:51:00
ripgrep       14.1.0                   main   2024-01-09 20:52:55
ruby          3.3.4-1                  main   2024-07-12 11:28:59
rust-analyzer 2024-08-05               main   2024-08-05 16:54:10
scoop-search  1.5.0                    main   2024-07-10 10:40:35
sd            1.0.0                    main   2023-12-09 16:55:11
starship      1.20.1                   main   2024-07-27 22:42:34
tealdeer      1.6.1                    main   2023-12-09 16:26:07
tokei         12.1.2                   main   2023-11-29 20:46:57
tree-sitter   0.22.6                   main   2024-05-10 10:55:49
unar          1.8.1                    main   2023-12-27 16:15:44
wezterm       20240203-110809-5046fc22 extras 2024-03-06 17:29:06
wget          1.21.4                   main   2023-11-29 22:29:31
which         2.20                     main   2023-12-02 14:27:18
whkd          0.2.1                    extras 2024-04-11 16:23:03
yarn          1.22.22                  main   2024-05-25 22:07:31
zoxide        0.9.4                    main   2024-02-22 12:17:08
```

## 安装 wezterm

wezterm是一个强大的跨平台终端模拟器，采用rust编写，在这个赛道有多个竞品，如alacritty，windows terminal，其中WT是json格式的config，alacritty作者摆烂，相比之下wezterm的作者就很勤快了，更重要的一个原因，wezterm是目前少数可以绕过windows下的ConPTY的terminal，它采用ssh的特性，使得wsl内可以实现很多原生linux才有的功能。

```bash
scoop install wezterm
```

配置文件参考[wezterm](https://github.com/marigold0225/wezterm)，位于~/.config/wezterm
![](wezterm.png)

## powershell 配置
在使用下面的配置文件之前，至少准备这些软件，都可以通过scoop安装：
- starship # rust 构建的shell外观美化
- yazi # rust 构建的终端文件管理器
- exa # rust 构建的系统工具，代替ls命令
- zoxide # rust 构建的系统工具，代替cd命令

也可以注释掉相应的部分来避免下载这些软件，接着打开powershell配置文件：notepad $PROFILE
```bash
Import-Module PSReadLine
Invoke-Expression (&starship init powershell)
Import-Module -Name Terminal-Icons

Set-PSReadLineOption -EditMode Emacs
Set-PSReadLineOption -PredictionSource HistoryAndPlugin
Set-PSReadLineOption -HistorySearchCursorMovesToEnd
Set-PSReadLineOption -HistorySaveStyle SaveIncrementally
Set-PSReadLineOption -PredictionViewStyle ListView
Set-PSReadLineOption -PredictionSource History
Set-PSReadLineOption -BellStyle None
Set-PSReadLineKeyHandler -Key 'Tab' -Function MenuComplete
Set-PSReadLineKeyHandler -Key 'Ctrl+z' -Function undo

# Set-PSReadLineKeyHandler -Key 'Tab' -Function Complete

Set-PSReadLineKeyHandler -Key 'Ctrl+k' -Function HistorySearchBackward
Set-PSReadLineKeyHandler -Key 'Ctrl+j' -Function HistorySearchForward

## open current folder

function OpenCurrentFolder {
param (
$Path = '.'
)
Invoke-Item $Path
}

## yazi

function yy {
$tmp = [System.IO.Path]::GetTempFileName()
    yazi $args --cwd-file="$tmp"
$cwd = Get-Content -Path $tmp
    if (-not [String]::IsNullOrEmpty($cwd) -and $cwd -ne $PWD.Path) {
Set-Location -LiteralPath $cwd
}
Remove-Item -Path $tmp
}

## k is a function to kill a process by its ID

function k {
Stop-Process -Id $args
}

## exa replace ls

function ls {
exa --icons --group-directories-first --color-scale --color=always --git --long --all --header --time-style=long-iso --group --classify $args
}

function l {
exa -lbah @args
}

function ll {
exa -lbg @args
}

function la {
exa -lbahg @args
}

## Set-Alias

Set-Alias -Name vi -Value nvim
Set-Alias -Name br -Value broot
Set-Alias -Name open -Value OpenCurrentFolder
Set-Alias -Name nide -Value neovide

Invoke-Expression (& { (zoxide init powershell | Out-String) })

```
# WSL2 CUDA
wsl2的安装不作介绍，两个概念：nvidia-smi是驱动,是nvidia显卡正常使用所必须的环境，nvcc是runtime，用于编译cpp和cu代码，NVIDIA官方建议在wsl2中无需安装任何cuda驱动，因为windows内安装完驱动之后，wsl内能自动识别到驱动：
```bash
which nvidia-smi
/usr/lib/wsl/lib/nvidia-smi

```
要在wsl内安装nvcc，则需要单独安装cuda-toolkit，但是一些linux发行版默认的包管理器安装cuda，会包含驱动，为了避免驱动覆盖，我们只需要单独安装cuda-toolkit，幸运的是，archlinux的pacman内的两个package均不含驱动，直接安装即可，如果是其他发行版，移步[NVIDIA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64)，选择特定的版本安装。
```bash
sudo pacman -S cuda cuda-tools
sudo pacman -Ss cuda
extra/cuda 12.5.1-1 [installed]
NVIDIA's GPU programming toolkit
extra/cuda-tools 12.5.1-1 [installed]
NVIDIA's GPU programming toolkit (extra tools: nvvp, nsight)
```
该安装可能会额外下载一个gcc-13或者其他版本，因为nvcc也对gcc版本有要求，这时候如果我们默认的gcc版本不兼容，我们需要指定环境变量cuda_host_CXX：
```bash
# nvidia hpc-sdk 用来编译fortran, 需要单独下载，搜索hpc-sdk
export NVARCH=`uname -s`_`uname -m`
export NVCOMPILERS=/opt/nvidia/hpc_sdk
export MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/24.7/compilers/man
export PATH=$NVCOMPILERS/$NVARCH/24.7/compilers/bin:$PATH
export PATH=$NVCOMPILERS/$NVARCH/24.7/comm_libs/mpi/bin:$PATH
export MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/24.7/comm_libs/mpi/man
export MODULEPATH=$NVCOMPILERS/modulefiles:$MODULEPATH

## cuda
export CUDADIR=/opt/cuda
export PATH=$CUDADIR/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDADIR/lib
export CPLUS_INCLUDE_PATH=$CUDADIR/include:$CPLUS_INCLUDE_PATH
export CUDA_HOME=/opt/cuda
## cuda_host_CXX
export NVCC-CCBIN=/usr/bin/g++-13
```

以上hpc-sdk和cuda都内置了nvcc，且版本一致，那为什么要下载两个呢？首先nvcc虽然看上去是一个cuda编译器，但是它实际上是一个驱动，用于调用g++编译cpp代码并且和cu代码链接起来，形成最终的可执行程序。它本身并不编译cpp，并且cuda toolkit内置了一系列cuda库，包括cuBLAS，cuFFT，cuDNN，TensorRT，一系列调试和性能分析工具，多用于机器学习，数据分析和科学计算中，而hpc-sdk是包含了CUDA的部分componet以及一系列编译器，nvfortran(没错，我就是为了这个下载的，并且hpc-sdk不支持windows)，nvc，nvc++，这个nvc++就可以直接编译cu文件。
hpc-sdk内置的编译器：
```bash
❯ which nvcc                          
/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvcc
❯ nvc -V                              

nvc 24.7-0 64-bit target on x86-64 Linux -tp alderlake 
NVIDIA Compilers and Tools
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
❯ nvc++ -V

nvc++ 24.7-0 64-bit target on x86-64 Linux -tp alderlake 
NVIDIA Compilers and Tools
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
❯ nvcc -V                   
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_02:18:23_PDT_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
❯ nvfortran -V                         

nvfortran 24.7-0 64-bit target on x86-64 Linux -tp alderlake 
NVIDIA Compilers and Tools
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
```
看上去hpc-sdk的编译器(nvcc,nvc,nvc++,nvfortran)比cuda的编译器(nvcc)完整，但是cuda的头文件比hpc-sdk内置cuda头文件更完整：
```bash
## 在cuda和hpc-sdk的安装目录下，我们分别查看头文件库，可以发现数量是不一致的，比如curand.h这个头文件hpc-sdk就没有:
❯ pwd
/opt/cuda/include
❯ ls | wc -l
156
❯ ls | rg cur
curand.h
curand_discrete.h
curand_discrete2.h
curand_globals.h
curand_kernel.h
curand_lognormal.h
curand_mrg32k3a.h
curand_mtgp32.h
curand_mtgp32_host.h
curand_mtgp32_kernel.h
curand_mtgp32dc_p_11213.h
curand_normal.h
curand_normal_static.h
curand_philox4x32_x.h
curand_poisson.h
curand_precalc.h
curand_uniform.h
❯ pwd
/opt/nvidia/hpc_sdk/Linux_x86_64/2024/cuda/include
❯ ls | wc -l 
119
❯ ls | rg cur
```
所以一般来说只需要下载cuda就够了(除非需要用到特定的编译器)，上面hpc-sdk的部分可以先注释掉

## 测试代码
首先是CMakeLists:
```cmake
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(test CUDA)

set(CMAKE_CUDA_STANDARD 20)
add_executable(test main.cu)

set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```
main.cu:
```cpp
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void cuda_hello() { printf("hello world from GPU!\n"); }

auto main() -> int
{
    cuda_hello<<<1, 10>>>();
    cudaDeviceReset();

    return 0;
}
```
首先，如果没有环境变量NVCC-CCBIN=/usr/bin/g++-13，直接使用nvcc编译，你可能会得到错误:

```bash
❯ nvcc main.cu 
In file included from /opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/include/cuda_runtime.h:82,
                 from <command-line>:
/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/include/crt/host_config.h:143:2: error: #error -
- unsupported GNU version! gcc versions later than 13 are not supported! The nvcc flag '-allow-u
nsupported-compiler' can be used to override this version check; however, using an unsupported h
ost compiler may cause compilation failure or incorrect run time execution. Use at your own risk
.
  143 | #error -- unsupported GNU version! gcc versions later than 13 are not supported! The nvc
c flag '-allow-unsupported-compiler' can be used to override this version check; however, using 
an unsupported host compiler may cause compilation failure or incorrect run time execution. Use 
at your own risk.

```
大概意思就是gcc版本不对，此时我们加上参数-ccbin gcc-13，这个gcc-13是下载cuda时自带的:

```bash
 nvcc -ccbin gcc-13 main.cu
 ./a.out
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
```
也可以直接用nvc++编译(如果下载的有hpc-sdk的话)：
```bash
❯ nvc++ main.cu
❯ ./a.out
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
```
如果在.cu的程序内包含了c++的标准库(比如各种STL)，需要指定c++链接库，因为nvcc并不处理c++的依赖：
```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
__global__ void cuda_hello() { printf("hello world from GPU!\n"); }

auto main() -> int
{
    cuda_hello<<<1, 10>>>();
    cudaDeviceReset();

    std::cout << "Hello world from CPU use STL!" << std::endl;

    return 0;
}
```
此时继续编译会报错：
```bash
❯ nvcc -ccbin gcc-13 main.cu
/usr/sbin/ld: /tmp/tmpxft_0001c412_00000000-11_main.o: warning: relocation again
st `_ZSt4cout' in read-only section `.text'
/usr/sbin/ld: /tmp/tmpxft_0001c412_00000000-11_main.o: in function `main':
tmpxft_0001c412_00000000-6_main.cudafe1.cpp:(.text+0xa6): undefined reference to
 `std::cout'
/usr/sbin/ld: tmpxft_0001c412_00000000-6_main.cudafe1.cpp:(.text+0xae): undefine
d reference to `std::basic_ostream<char, std::char_traits<char> >& std::operator
<< <std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&,
 char const*)'
/usr/sbin/ld: tmpxft_0001c412_00000000-6_main.cudafe1.cpp:(.text+0xb5): undefine
d reference to `std::basic_ostream<char, std::char_traits<char> >& std::endl<cha
r, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)'
/usr/sbin/ld: tmpxft_0001c412_00000000-6_main.cudafe1.cpp:(.text+0xc0): undefine
d reference to `std::ostream::operator<<(std::ostream& (*)(std::ostream&))'
/usr/sbin/ld: warning: creating DT_TEXTREL in a PIE
collect2: error: ld returned 1 exit status
```
加上-lstdc++命令之后成功：
```bash
❯ nvcc -ccbin gcc-13 main.cu -lstdc++
❯ ./a.out
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
Hello world from CPU use STL!
```
所以最好用NVCC-CCBIN环境变量来指定host端的c++compiler，这样直接使用nvcc编译，也会自动链接到对应的库
如果使用cmake编译：
```bash
 cmake .. && make -j
-- The CUDA compiler identification is NVIDIA 12.5.82
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /opt/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Configuring done (3.7s)
-- Generating done (0.0s)
-- Build files have been written to: /home/mafu/Documents/cuda/0/build
[ 33%] Building CUDA object CMakeFiles/test.dir/main.cu.o
[ 66%] Linking CUDA device code CMakeFiles/test.dir/cmake_device_link.o
[100%] Linking CUDA executable test
[100%] Built target test
 ./test
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
hello world from GPU!
```
（同样，如果包含c++标准库请在CMakelists.txt末尾添加: target_link_libraries(test stdc++)
但是假如我们把NVCC-CCBIN变量从zshrc内移除：
```bash
 echo $NVCC-CCBIN (无输出)

 cmake .. && make -j
CMake Error at /usr/share/cmake/Modules/CMakeDetermineCompilerId.cmake:838 (message):
  Compiling the CUDA compiler identification source file
  "CMakeCUDACompilerId.cu" failed.

  Compiler: /opt/cuda/bin/nvcc

  Build flags:

  Id flags: --keep;--keep-dir;tmp -v
```
一大堆看不懂的错误，此时要想正常编译，需要以下参数指定版本:
```bash
cmake -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/gcc-13" ..
```
总而言之，cmake貌似默认使用nvcc作为编译器，wsl下要想编译cu代码，请使用以下命令:
```bash
nvcc -ccbin gcc-13 main.cu -lstdc++
nvcc main.cu #(指定NVCC-CCBIN)
# or
nvc++ main.cu
# or cmake (with export NVCC-CCBIN=/usr/bin/g++-13)
cmake .. && make
# or without NVCC-CCBIN
cmake -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/gcc-13" ..
# 即注意nvcc支持的gcc版本以及c++标准库
```
## 测试代码2（c++和cuda混合编译）
CMakeLists
```cmake
cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(cuda_project CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_project main.cpp kernel1.cu kernel2.cu)

set_target_properties(cuda_project PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```
main.cpp
```cpp
#include <iostream>

void kernel1();
void kernel2();

auto main() -> int
{
    std::cout << "Calling kernel1" << std::endl;
    kernel1();

    std::cout << "Calling kernel2" << std::endl;
    kernel2();

    return 0;
}
```
kernel1
```cpp
#include <stdio.h>
__global__ void kernel1_func() { printf("Hello from kernel1\n"); }

void kernel1()
{
    kernel1_func<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```
kernel2
```cpp
#include <stdio.h>
__global__ void kernel2_func() { printf("Hello from kernel2\n"); }

void kernel2()
{
    kernel2_func<<<1, 1>>>();
    cudaDeviceSynchronize();
}
```
编译
```cmake
❯ cmake .. && make -j
-- The CUDA compiler identification is NVIDIA 12.5.82
-- The CXX compiler identification is GNU 14.2.1
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /opt/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/sbin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Configuring done (4.7s)
-- Generating done (0.0s)
-- Build files have been written to: /home/mafu/Documents/cuda/test/build
[ 40%] Building CUDA object CMakeFiles/cuda_project.dir/kernel1.cu.o
[ 40%] Building CXX object CMakeFiles/cuda_project.dir/main.cpp.o
[ 60%] Building CUDA object CMakeFiles/cuda_project.dir/kernel2.cu.o
[ 80%] Linking CUDA device code CMakeFiles/cuda_project.dir/cmake_device_link.o
[100%] Linking CXX executable cuda_project
[100%] Built target cuda_project
❯ ./cuda_project
Calling kernel1
Hello from kernel1
Calling kernel2
Hello from kernel2
```
另外，可能你已经发现了cmake的输出信息，显示识别到CXX编译器版本是GNU 14.2.1，并不是gcc-13，这是为什么呢？因为我们的环境变量里指定了export NVCC-CCBIN=/usr/bin/g++-13，虽然识别到gcc14，但是编译还是使用的gcc13，不信可以将这个变量取消验证，cmake会报错的，提示没有set cxx编译器。

