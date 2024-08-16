---
title: latex-tikz
date: 2021-11-28 14:29:49
tags:
	- latex 
catecategories: latex
mathjax: true
---
latex下tikz作图
<!--more-->
## 在线编辑
直接给两个网址[tikzcd-editor](https://tikzcd.yichuanshen.de/),[latex-preview](http://www.tlhiv.org/ltxpreview/).
## 一些例子
展示下tikz能做的图
```
\begin{tikzpicture}[domain=0:4]
  \draw[very thin,color=gray] (-0.1,-1.1) grid (3.9,3.9);
  \draw[->] (-0.2,0) -- (4.2,0) node[right] {$x$};
  \draw[->] (0,-1.2) -- (0,4.2) node[above] {$f(x)$};
  \draw[color=red]    plot (\x,\x)             node[right] {$f(x) =x$};
  \draw[color=blue]   plot (\x,{sin(\x r)})    node[right] {$f(x) = \sin x$};
  \draw[color=orange] plot (\x,{0.05*exp(\x)}) node[right] {$f(x) = \frac{1}{20} \mathrm e^x$};
\end{tikzpicture}
```
![](1.svg)
```
\usetikzlibrary{decorations.pathmorphing}
\begin{tikzpicture}[line width=0.2mm,scale=1.0545]\small
\tikzset{>=stealth}
\tikzset{snake it/.style={->,semithick,
decoration={snake,amplitude=.3mm,segment length=2.5mm,post length=0.9mm},decorate}}
\def\h{3}
\def\d{0.2}
\def\ww{1.4}
\def\w{1+\ww}
\def\p{1.5}
\def\r{0.7}
\coordinate[label=below:$A_1$] (A1) at (\ww,\p);
\coordinate[label=above:$B_1$] (B1) at (\ww,\p+\h);
\coordinate[label=below:$A_2$] (A2) at (\w,\p);
\coordinate[label=above:$B_2$] (B2) at (\w,\p+\h);
\coordinate[label=left:$C$] (C1) at (0,0);
\coordinate[label=left:$D$] (D) at (0,\h);
\draw[fill=blue!14](A2)--(B2)-- ++(\d,0)-- ++(0,-\h)--cycle;
\draw[gray,thin](C1)-- +(\w+\d,0);
\draw[dashed,gray,fill=blue!5](A1)-- (B1)-- ++(\d,0)-- ++(0,-\h)-- cycle;
\draw[dashed,line width=0.14mm](A1)--(C1)--(D)--(B1);
\draw[snake it](C1)--(A2) node[pos=0.6,below] {$c\Delta t$};
\draw[->,semithick](\ww,\p+0.44*\h)-- +(\w-\ww,0) node[pos=0.6,above] {$v\Delta t$};
\draw[snake it](D)--(B2);
\draw[thin](\r,0) arc (0:atan2(\p,\w):\r) node[midway,right,yshift=0.06cm] {$\theta$};
\draw[opacity=0](-0.40,-0.14)-- ++(0,5.06);
\end{tikzpicture}
```
![](2.svg)
```
\begin{tikzcd} T
\arrow[drr, bend left, "x"]
\arrow[ddr, bend right, "y"]
\arrow[dr, dotted, "{(x,y)}"] & & \\
& X \times_Z Y \arrow[r, "p"] \arrow[d, "q"]& X \arrow[d, "f"] \\
& Y \arrow[r, "g"]& Z
\end{tikzcd}
```
![](5.svg)
甚至！

```
\usetikzlibrary{positioning, arrows.meta}
\tikzset{
  rect1/.style = {
    shape = rectangle,
    draw = green,
    text width = 3cm,
    align = center,
    minimum height = 1cm,
  }
}
\tikzset{
  arrow1/.style = {
    draw = purple, thick, -{Latex[length = 4mm, width = 1.5mm]},
  }
}
\tikzset{
  arrow2/.style = {
    draw = purple, thick, {Latex[length = 4mm, width = 1.5mm]}-{Latex[length = 4mm, width = 1.5mm]},
  }
}
\begin{center}
  \begin{tikzpicture}
    \node[rect1, fill = green!60!white](lexical){lexical analysis};
    \node[rect1, fill = green!40!white, below = of lexical](Gramma){Gramma analysis};
    \node[rect1, fill = green!20!white, below = of Gramma, text width = 5cm](Semantic middle){Semantic Analysis、Intermediate representation};
    \node[rect1, fill = green!60!black, below = of Semantic middle](optimization){\color{white}Code optimization};
    \node[rect1, fill = green!30!black, below = of optimization](generate){\color{white}Code generation};
    \node[rectangle, fill = red!20!white, draw = red, text width = 2.0cm, minimum height = 4cm, align = center, left = 2cm of Semantic middle](Symbol){Symbol table management};
    \node[rectangle, fill = blue!20!white, draw = blue, text width = 2.0cm, minimum height = 4cm, align = center, right = 2cm of Semantic middle](Error handling){Error handling};
    \draw[arrow1](0, 50pt)node[right, yshift = -15pt]{Input source program} -- (lexical);
    \draw[arrow1](lexical) -- node[right]{Word flow}(Gramma);
    \draw[arrow1](Gramma) -- node[right]{grammar}(Semantic middle);
    \draw[arrow1](Semantic middle) -- node[right]{Middle representation}(optimization);
    \draw[arrow1](optimization) -- node[right]{Middle representation after optimization}(generate);
    \draw[arrow1](generate) -- node[right, yshift = 5pt, xshift = 5pt]{Object code}++(0, -50pt);
    \draw[arrow2](Symbol) -- (lexical.west);
    \draw[arrow2](Symbol) -- (Gramma.west);
    \draw[arrow2](Symbol) -- (Semantic middle);
    \draw[arrow2](Symbol) -- (optimization.west);
    \draw[arrow2](Symbol) -- (generate.west);
    \draw[arrow2](Error handling) -- (lexical.east);
    \draw[arrow2](Error handling) -- (Gramma.east);
    \draw[arrow2](Error handling) -- (Semantic middle);
    \draw[arrow2](Error handling) -- (optimization.east);
    \draw[arrow2](Error handling) -- (generate.east);
    \draw[thick, dashed, draw = purple](-125pt, 27pt)node[below right]{Compiler front end} -- (125pt, 27pt) -- (125pt, -250pt) -- (-125pt, -250pt)node[above right]{Compiler backend} -- (-125pt, 27pt);
    \draw[thick, dashed, draw = purple](-125pt, -135pt) -- (125pt, -135pt);
  \end{tikzpicture}
  \\ fig.1 Compiler structure diagram
\end{center}
```
![](3.svg)
```
\usetikzlibrary{positioning, arrows.meta, calc}
\tikzset{
  arrow1/.style = {
    draw = purple, thick, -{Latex[length = 4mm, width = 1.5mm]},
  }
}
\tikzset{
  nonterminal/.style = {
    rectangle,
    align = center,
    minimum size = 6mm,
    very thick,
    draw = red!50!black!50,
    top color = white,
    bottom color = red!50!black!20,
  }
}
\tikzset{
  terminal/.style = {
    rectangle,
    align = center,
    minimum size = 6mm,
    rounded corners = 3mm,
    very thick,
    draw = black!50,
    top color = white,
    bottom color = black!20,
  }
}
\hspace{-3cm}{
\begin{minipage}{15.4cm}
  \begin{tikzpicture}[node distance = 0.7cm]
    \node[terminal](function){FUNCTION};
    \node[nonterminal, right = of function](name1){Function name};
    \node[terminal, right = of name1](left){(};
    \node[terminal, right = of left](var){VAR};
    \node[nonterminal, right = of var](para){Formal parameters};
    \node[terminal, right = of para](colon1){:};
    \node[nonterminal, right = of colon1](type1){Type name};
    \node[terminal, right = of type1](right){)};
    \node[terminal, right = of right](colon2){:};
    \node[nonterminal, right = of colon2](type2){Type name};
    \node[terminal, right = of type2](semicolon1){;};
    \node[terminal, below = of para](semicolon2){;};
% ----------------------------------------------------------
    \draw[arrow1](function)--(name1);
    \draw[arrow1](name1)--(left);
    \draw[arrow1](left)--(var);
    \draw[arrow1](var)--(para);
    \draw[arrow1](para)--(colon1);
    \draw[arrow1](colon1)--(type1);
    \draw[arrow1](type1)--(right);
    \draw[arrow1](right)--(colon2);
    \draw[arrow1](colon2)--(type2);
    \draw[arrow1](type2)--(semicolon1);
    \draw[arrow1]($ (var.west)+(-3mm, 0) $)--++(0, 6mm)-|($ (var.east)+(3mm, 0) $);
    \draw[arrow1]($ (type1.east)+(3mm, 0) $)|-(semicolon2);
    \draw[arrow1](semicolon2)-|($ (var.west)-(3mm, 0) $);
    \draw[arrow1]($ (name1.east)+(3mm, 0) $)--++(0, -20mm)-|($ (right.east)+(3mm, 0) $);
% ----------------------------------------------------------
    \node[nonterminal, below = 2.7cm of function, xshift = 1cm](description){Description section};
    \node[terminal, right = of description](begin){BEGIN};
    \node[nonterminal, right = of begin, text width = 1.6cm](statement){Statement};
    \node[terminal, right = of statement](semicolon3){;};
    \node[terminal, right = of semicolon3, text width = 1.6cm](end){END};
    \node[terminal, right = of end](semicolon4){;};
% ----------------------------------------------------------
    \draw[arrow1](description)--(begin);
    \draw[arrow1](begin)--(statement);
    \draw[arrow1](statement)--(semicolon3);
    \draw[arrow1](semicolon3)--(end);
    \draw[arrow1](end)--(semicolon4);
    \draw[arrow1](semicolon4)--++(30pt, 0);
    \draw[arrow1](semicolon1)--++(0, -2.5cm)-|(description);
    \draw[arrow1]($ (semicolon3.east)+(3mm, 0) $)--++(0, -6mm)-|($ (begin.east)+(3mm, 0) $);
  \end{tikzpicture}
\end{minipage}
```
![](4.svg)
## 结尾
画简单的流程图，函数图和费曼图没什么问题，但是数据做图还是老老实实用作图软件.

