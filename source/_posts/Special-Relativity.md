---
title: General_Relativity
date: 2021-07-06 14:42:15
tags:
    - General_Relativity
categories: 公式推导
mathjax: true
    - 公式推导
---
关于广义相对论的一点数学知识和概念
<!--more-->
## 等效原理
弱等效原理：引力场与惯性场的力学效应是局域不可区分的
强等效原理：引力场和惯性场的一切物理效应是局域不可区分的
惯性系：在引力场中自由下落的无自转的无穷小的参考系
## 黎曼几何和张量分析
### 在SR下的定义 （lorentz变化）
$$x^{\prime}_{\mu}=a_{\mu \nu}x_{\nu}$$  
$$dx_{\nu}=(a^{-1})_{\nu \mu}dx^{\prime}_{\mu}$$   
$aa^{-1}=I$   (坐标变化线性正交) <br />    
标量：坐标变化下不变<br />      
矢量：坐标变化下和坐标微分元一样变的量.<br />    
$${V^{\prime}_{\mu}=a_{\mu \nu}V_{\nu}}$$
$${dV^{\prime}_{\mu}=a_{\mu \nu}dV_{\nu}}$$     
张量：坐标变化下,按如下变化的：      
$$T^{\prime}_{\mu \nu}=a_{\mu \alpha}a_{\nu \beta}T_{\alpha \beta}$$   
### 在GR下的定义（广义坐标变化）
$x^{\prime \mu}=x^{\prime \mu}(x^{\nu})$  ,坐标变化既不正交也不线性    
$$dx^{\prime \mu}=\frac{\partial x^{\prime \mu}}{\partial x^{\nu}}dx^{\nu}$$   
标量：在广义坐标变化下不变的量   
矢量：在广义坐标变化下和坐标微分元一样变的量（逆变和协变之分）      
张量：$T^{\prime \mu \nu}=\frac{\partial x^{\prime \mu}}{\partial x^{\alpha}}\frac{\partial x^{\prime \nu}}{\partial x^{\beta}}T^{\alpha \beta}$   (逆变协变之分)<br />   
克罗内克尔符号：二阶混合张量（证明）   
### 张量代数
缩并,对称性
## 平移和联络
联络定义矢量的平移（计算）
i）联络不是张量
ii）两个联络之差是张量
iii）联络的反对称部分是张量
## 协变微商
标量场的协变微商就是普通微商,且结果是逆变矢量
协变矢量的协变微商：$$A_{\mu ; \nu}=A_{\mu , \nu}-\Gamma^{\lambda}_{\mu \nu}A_{\lambda}$$<br />   
逆变矢量的协变微商：$$A^{\mu}_{ ; \nu}=A^{\mu}_{ , \nu}+\Gamma^{\mu}_{\lambda \nu}A^{\lambda}$$<br />  
再用莱布尼兹可以算出其他张量的微商
## 测地线和仿射参量
曲线的参数方程：$x^{\mu}=x^{\mu}(\lambda)$<br />   
切矢：$A^{\mu}=\frac{dx^{\mu}}{d \lambda}$<br />   
要求矢量平移后方向相同,可以得到测地线方程：   
$\frac{d^{2}x^{\mu}}{d\sigma^{2}}+\Gamma^{\mu}_{\alpha \beta}\frac{dx^{\alpha}}{d\sigma}\frac{dx^{\beta}}{d\sigma}=0$<br />    
曲线上两点平移后切矢量重合,该曲线是测地线。
## 曲率与绕率
两者之差：$$A_{\lambda ; \mu ; \nu}-A_{\lambda ; \nu ; \mu}=R^{\rho}_{\lambda \mu ; \nu}A_{\rho}-2\Gamma^{\rho}_{[\nu \mu]}A_{\lambda ; \rho}$$   
前者曲率,后者绕率   
绕率：矢量平移一周后不能形成封闭空间,空间是扭曲的   
曲率：矢量平移一周后会有角度差   
### 曲率的性质
i）后一对指标反对称
ii）两种独立的缩并方式（13缩并,12缩并）
## 度规和距离
平直时空：$g_{\mu \nu}$ 是对角元-1和1<br />    
度规能升降指标    
矢量的平移：计算联络       
平移切矢方向不变：计算出测地线           
平移长度不变：计算联络（克式符）        
$$\Gamma^{\alpha}_{\mu \nu}=\frac{1}{2}g^{\alpha \lambda}(g_{\mu \lambda , \nu}+g_{\nu \lambda , \mu}-g_{\mu \nu , \lambda})$$      
度规的协变微商等于0      
在一个无绕空间,总能找到一个坐标变化,将时空任意一点的克式符变为0      
短程线：对两点曲线做变分,计算欧拉拉格朗日方程      
$$\frac{d^{2}x^{\mu}}{ds^{2}}+\Gamma^{\mu}_{\alpha \beta}\frac{dx^{\alpha}}{ds}\frac{dx^{\beta}}{ds}=0$$  
## 黎曼时空的曲率张量
$$R^{\rho}_{\lambda [\mu \nu]}$$
i）后一对指标反对称
ii）存在两种缩并
克式符下的曲率张量的对称性
$$R_{\rho \lambda \mu \nu}=g_{\rho \sigma}R^{\sigma}_{\lambda \mu \nu}$$   
i)后一对指标反对称
ii）前一对指标反对称
iii）前一对指标与后一对指标对称
iiii）里奇恒等式：$$R^{\rho}_{\lambda \mu \nu}+R^{\rho}_{\mu \nu \lambda}+R^{\sigma}_{\nu \lambda \mu}=0$$(后三个指标轮转)<br />
三个重要派生张量：
里奇张量：$$R_{\mu \nu}$$  (缩并得到)<br />   
曲率标量：R（再次缩并）<br />   
爱因斯坦张量：$$G_{\mu \nu}=R_{\mu \nu}-\frac{1}{2}R$$<br />   
毕安基恒等式：（后三个指标轮转的协变微商）<br />
$$G^{\mu \nu}_{ ; \nu}=0$$
$$(R_{\mu \nu}-\frac{1}{2}R)^{ ; \nu}=0$$
## 重要运算
度规的微分,广义相对论就是算度规的一阶导数和二阶导数（度规-联络-曲率）<br />
特殊的克式符：$$\Gamma^{\mu}_{\alpha \mu}=\frac{\partial}{\partial x^{\alpha}}(\ln \sqrt{-g})$$<br />
散度运算：$$div A^{\mu}=A^{\mu}_{ ; \mu}=\frac{1}{\sqrt{-g}}\frac{\partial}{\partial x^{\mu}}(\sqrt{-g}A^{\mu})$$<br />   
达郎贝尔算符:$$\Box=div(grad\Psi)$$<br />   
旋度运算:$$A_{\mu ; \nu}-A_{\nu ; \mu}=A_{\mu , \nu}-A_{\nu , \mu}$$
## 爱因斯坦方程
坐标时和固有时的区别
坐标时：$t=\frac{x^{0}}{c}$,没有意义,用与计算和逻辑分析 <br />   
固有时：$$ds^{2}=g_{\mu \nu}dx^{\mu}dx^{\nu}$$,$dT=\frac{ids}{c}=d \tau$<br />  
两者联系：$$d\tau=\sqrt{-g_{00}}dt$$<br />   
这也是地球太阳时间间隔区别,太阳附近要乘以太阳的度规：$${g_{00}}$$<br />   
能量动量张量：   
电磁场：$$T^{\mu \nu}=\frac{1}{4\pi}(F^{\mu}_{\lambda}F^{\lambda \nu}-\frac{1}{4}g^{\mu \nu}F_{\rho \lambda}F^{\rho \lambda})$$<br />   
理想流体：$T^{\mu \nu}=(\rho+\frac{P}{c^{2}})u^{\mu}u^{\nu}+Pg^{\mu \nu}$<br />   
松散介质,非相对论理想流体：$T^{\mu \nu}=\rho u^{\mu}u^{\nu}$<br />   
坐标条件和边界条件:<br />   
协和坐标系：$g^{\mu \nu}\Gamma^{\lambda}_{\mu \nu}=0$   
## 应用
### 史瓦西解
静态球对称物体,解出度规：
$$ds^{2}=-c^{2}(1-\frac{2GM}{c^{2}r})dt^{2}+(1-\frac{2GM}{c^{2}r})^{-1}dr^{2}+r^{2}d\theta^{2}+r^{2}sin^{2}\theta d\phi^{2}$$   
地球上的钟:$$d\tau_{2}$$ <br />
太阳上的钟:$$d\tau_{1}$$ <br />
两者差距$g_{00}$:
$$d\tau_{2}=(1-\frac{2GM}{c^{2}r})^{-1/2}d\tau_{2}$$
引力红移: $$v=v_{0}\sqrt{1-\frac{2GM}{c^{2}r}}$$
水星进动
史瓦西黑洞：静态球对称
克尔纽曼黑洞：带电旋转