# svm

学渣的svm简单推导过程

1 线性可分svm，找到一个超平面将样本分为正负两类

2 概念：函数间隔$\hat\gamma$: $y(wx+b)$

3 概念：几何间隔$\gamma$: 由于w和b成比例的变化后，分离平面还是同一个，但是函数间隔却是成倍增加了，因此对w加约束，即使让它除以模长，得到几何间隔 $y(\frac{w}{||w||} x + \frac{w}{||w||}b)$，若||w||为1，则函数间隔=几何间隔    (这里其实也可以直接用点到直线的距离推出几何间隔，省略第2步)。

3.5 可知$\gamma = \frac{\hat\gamma}{||w||}$,并且由这个公式可以确认，w成倍增加后，$\hat\gamma$变化,$\gamma不变$（因为总归要除以模长的)。

4 设立目标：找到一个几何间隔 $\gamma$ ,使得每个 $\gamma_i>= \gamma$ ，并且要令这个$\gamma$最大化（支持向量与分离超平面最大化）:
$\max\limits_{w,b}\qquad \gamma$
$s.t. \qquad y_i(\frac{w}{||w||} x_i + \frac{w}{||w||}b）>=\gamma,i = 1,2,3...N$

5 将上式转变为
$\max\limits_{w,b}\qquad \frac{\hat\gamma}{||w||}$ （3.5得到）
$s.t. \qquad  y_i(wx_i+b)>=\hat\gamma,i = 1,2,3...N$  （左右两边同乘||w||）

6 固定$\hat\gamma$（应该可以轻松的看到，$\hat\gamma$不影响求解，因此5变为：
$\max\limits_{w,b}\qquad \frac{1}{||w||}$ 
$s.t. \qquad  y_i(wx_i+b)>=1,i = 1,2,3...N$ 

7 将6转为(这也是个很简单的转化)：
$\min\limits_{w,b}\qquad \frac{1}{2}||w||^2$ 
$s.t. \qquad  y_i(wx_i+b)>=1,i = 1,2,3...N$ 

8 求解不等式约束，用拉格朗日法，将每个不等式都引入拉格朗日乘子$\alpha$，定义拉格朗日函数为：
$L(w,b,\alpha)=\frac{1}{2}||w||^2-\sum^{N}_{i=1}\alpha_iy_i(wx_i+b) + \sum^{N}_{i=1}\alpha_i$

此时，最优化问题为：

$\min\limits_{w,b}L(w,b,\alpha),\alpha >=0$，记为P

然后令Q问题为
$\min\limits_{w,b}\max\limits_{\alpha}L(w,b,\alpha)$

Q和P问题等价，（证明来源于广义拉格朗日函数的极小极大问题，大概意思是只要$\alpha$满足约束条件，那Q和P就是等价的，稍微超出知识体系了。）

此时，也是可以直接用梯度下降或者其他方法求解的，就是固定w和b,迭代$\alpha$,再迭代w和b

9 8的$min max$问题的对偶问题为:
$\max\limits_{\alpha}\min\limits_{w,b}L(w,b,\alpha)$,他们的关系为
$\max\limits_{\alpha}\min\limits_{w,b}L(w,b,\alpha) <= \min\limits_{w,b}\max\limits_{\alpha}L(w,b,\alpha)$,当满足5个kkt条件时，等号成立（超出了知识体系）

此时，先固定$\alpha$，对$w和b$分别求导数=0，得到：
$w=\sum^{N}_{i=1}\alpha_iy_ix_i$

$\sum^{N}_{i=1}\alpha_iy_i =0$,
代入原式，得到对偶问题的求解转为求：
$\max\limits_{\alpha} -\frac{1}{2}\sum^N_i\sum^N_j\alpha_i\alpha_jy_iy_j(x_ix_j)+\sum^N_i\alpha_i$,
$s.t. \qquad \sum^N_i\alpha_iy_i=0, \alpha_i>=0,i=1,2....N$

10 线性不可分（弱）：加入松弛变量,将7的问题转为：
$\min\limits_{w,b}\qquad \frac{1}{2}||w||^2 + C\sum^N_ik_i$
$s.t. \qquad  y_i(wx_i+b)>=1 - k_i,i = 1,2,3...N$
其中C为惩罚参数，越大惩罚越大，K为松弛变量。

11 线性不可分（弱）的对偶问题求解为：

$\max\limits_{\alpha} -\frac{1}{2}\sum^N_i\sum^N_j\alpha_i\alpha_jy_iy_j(x_ix_j)+\sum^N_i\alpha_i$,
$s.t. \qquad \sum^N_i\alpha_iy_i=0, 0<=\alpha_i<=C_i,i=1,2....N$
(所以和线性可分的svm差不多的)

12 smo求解

简单的来说，就是固定其他的$\alpha$,找到其中的两个最不符合kkt条件的$\alpha_i和\alpha_j$,两两更新（因为$\sum\alpha_iy_i=0$,改变其中一个，必定要改变另一个,具体的求解过程可以看我的代码实现。

13 线性不可分（强），引入核函数K和映射函数Q。（超出知识体系了）
首先回顾一下之前的svm，本质上，用对偶问题求解的svm，最终会留下几个支持向量，和这些支持向量的$\alpha$，其他样本中非支持向量的$\alpha=0$，并且求解和预测的时候都会计算特征之前的内积$x_ix_j$,而核函数K和映射函数Q的关系为：
$k(xi,xj) = q(xi) * q(xj)$,Q函数负责将特征映射到高维特征空间中，而核函数K则表示高维特征空间的两个向量内积，引入核函数可以隐式的学习到高维特征空间的特征，又很方便的在smo算法中将向量内积$xi * xj$替换为$k(xi, xj)$。
