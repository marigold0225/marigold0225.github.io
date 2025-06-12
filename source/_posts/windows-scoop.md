---
title: windows_scoop
date: 2021-11-23 21:25:19
tags:
	- scoop 
	- wsl
categories: linux
---
Windows ternimal 配置
<!--more-->
# scoop
## 安装windows terminal
直接去microsoft store下载
## 安装powershell7
[powershell7](https://github.com/PowerShell/PowerShell/releases)选择win-x64-msi安装
## scoop
获取权限
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
下载
```
iwr -useb get.scoop.sh | iex
```
常用命令
```
scoop update #更新
scoop config set proxy 127.0.0.1:7890 #设置代理
scoop config rm proxy #删除代理
# bucket 源
scoop bucket add main # 默认
scoop bucket add extras # 推荐
scoop bucket add versions
scoop bucket add nightlies
scoop bucket add nirsoft
scoop bucket add php
scoop bucket add nerd-fonts
scoop bucket add nonportable
scoop bucket add java
scoop bucket add games
scoop bucket add jetbrains # 推荐
# 常用软件
scoop install aria2
scoop install vim
scoop install python
scoop install go
scoop install vscode
scoop install sublime-text
scoop install wget
```
## oh my posh 安装
安装
```
Set-ExecutionPolicy Bypass
Install-Module oh-my-posh -Scope CurrentUser
Install-Module posh-git -Scope CurrentUser
```
查看主题
```
$PROFILE
if (!(Test-Path -Path $PROFILE )) { New-Item -Type File -Path $PROFILE -Force }
notepad $PROFILE

#在打开的文件中输入
Import-Module posh-git
Import-Module oh-my-posh
Set-Theme Honukai
```
# WSL2
```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

wsl --set-default-version 2
```
