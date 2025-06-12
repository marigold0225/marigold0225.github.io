---
title: tidal_deformability
date: 2021-11-22 19:16:38
tags:
	- neutron star 
	- General_Relativity
categories: 公式推导
mathjax: true
	- 公式推导
---
Tidal deformability of neutron star
<!--more-->
# various tidal love number of compact star
When spherical star placed in a static external quadrupolar tidal field $\varepsilon_{ij}$ then the star will be deformed and quadrupole deformation will be the leading order perturbation. Such a deformation is defined as the radio of the mass quadrupole moment of a star $Q_{ij}$ to the external tidal field $\varepsilon_{ij}$:
$$
\begin{equation}
\lambda=\frac{Q_{ij} } {\varepsilon_{ij} }
\end{equation}
$$
Specifically, the observable of the tidal deformability parameter $\lambda$ depends on the EOS via the neutron star(NS) radius and a dimensionless quantity $k_{2}$, called the Love number and is given by the relation:
$$
\begin{equation}
\lambda=\frac{2}{3}k_{2}R^{5},
\end{equation} 
$$
and the dimensionless tidal-deformability($\Lambda$) is related with the compactness parameter $C=M/R$ as:
$$
\begin{equation}
\lambda=\frac{2 k_{2} } {3 C^{5} },
\end{equation} 
$$
where R is the radius of the star in isolation. We have to get $k_{2}$ for the calculation of the deformability parameter $\lambda$.
To estimate the love number $k_{l}(l=2,3,4)$, along with the evaluation of the TOV equations, we have to compute $y=y_{l}(R)$ with initial boundary condition $y(0)=l$ from the following first order different equation iteratively:
$$
\begin{equation}
r\frac{dy(r)}{dr}+y(r)^{2}+y(r)F(r)+r^{2}Q(r)=0,
\end{equation}
$$
with,<br>
$$
\begin{equation}
F(r)=\frac{r-4\pi r^{3}[\varepsilon(r)-P(r)]}{r-2M(r)},
\end{equation}
$$
$$
\begin{equation}
Q(r)=\frac{4\pi(5\varepsilon(r)+9P(r)+\frac{\varepsilon(r)
+P(r)}{\partial P(r)/\varepsilon(r)}-\frac{l(l+1)}{4\pi r^{2}})}{r-2M(r)}- 
4[\frac{M(r)+4\pi r^{3}P(r)}{r^{2}(1-2M(r)/r)}]^{2}.
\end{equation}
$$
once,we know the value of $y=y_{l}(R)$, the love numbers $k_{l}$ are found from the following expression:
$$
\displaylines{k_{2}=\frac{8}{5}(1-2C)^{2}C^{5}[2C(y_{2}-1)-y_{2}+2]\{2C(4(y_{2}+1)C^{4}+(6y_{2}-4)C^{3}+ \\
(26-22y_{2})C^{2}+3(5y_{2}-8)C-3y_{2}+6)-3(1-2C)^{2}(2C(y_{2}-1)-y_{2}+2)\log(\frac{1}{1-2C})\}^{-1}}
$$
and $k_{3}, k_{4}$ ref from [Relativistic tidal properties of neutron stars](https://arxiv.org/pdf/0906.0096.pdf).
As we have emphasized earlier, the dimensionless love number $k_{l}(l=2,3,4)$ is an important quantity to measure the internal structure of the constituent body. These quantities directly enter into the gravitational wave phase of inspiralling binary neutron star and extract the information of the EOS. Notice that equation contain an overall factor $(1-2C)^{2}$, which tends to zero when the compactness approaches the compactness of the black hole $C^{BH}=1/2$. Also, it is to be pointed out that the presence of multiplication order factor $C$ with $(1-2C)^{2}$ in the expression of $k_{l}$ that the value of the love number of a black hole simply becomes zero $k_{l}^{BH}=0$.
## surficial love number
We calculate the surficial love number $h_{l}$ which describes the deformation of the body's surface in a multipole expansion. Damour and Nagar have given the surficial love number (also known as shape love number)$h_{l}$ for the coordinate displacement $\delta R$ of the body's surface under the external tidal force[surfical love number](https://arxiv.org/pdf/0906.0096.pdf) and Landry and Poisson have proposed the defination of Newtonian love number is terms of a curvature perturbation $\delta R$ instead of a surface displacement $\delta R$[Newtonian love number](https://arxiv.org/pdf/1404.6798.pdf). For a perfect fluid, the relation between the surficial love number $h_{l}$ and tidal love number $k_{l}$ is given as :
$$
\begin{equation}
h_{l}=\Gamma_{1}+2\Gamma_{2}k_{l}
\end{equation}
$$
$$
\begin{equation} 
\Gamma_{1}=\frac{l+1}{l-1}(1-C)F(-l,-l,-2l;2C)-\frac{2}{l-1}F(-l,-l,-1,-2l;2C) 
\end{equation}
$$
$$
\begin{equation}
\Gamma_{2}=\frac{1}{l+2}(1-C)F(l+1,l+1,2l+2;2C)+\frac{2}{l+2}F(l+1,l,2l+2;2C).
\end{equation}
$$
where $F(a,b,c;d)$ is the hypergeometric function .
## magnetic tidal love number
The quadrupolar case (l=2):
$$
\displaylines{j_{2}=\{96C^{5}(2C-1)(y_{2}-3)\}\{5(2C(12(y_{2}+1)C^{4} \\
+2(y_{2}-3)C^{3}+2(y_{2}-3)C^{2}+3(y_{2}-3)C- \\
3y_{2}+9)+3(2C-1)(y_{2}-3)\log(1-2C))\}^{-1}}
$$
