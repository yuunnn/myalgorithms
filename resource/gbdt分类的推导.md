泰勒展开：

$f(x+\Delta{x})= f(x) + f'(x)*\Delta{x}$ ,

损失函数：

$L(Fm,y)->L(Fm-1+fm,y)=L(Fm-1,y)+L'(Fm-1,y)*fm$

另目标函数为最小化损失函数的fm：

$c = L(Fm-1,y)+L'(Fm-1,y)*fm$

在m轮的cart树是一颗能够使得这个目标函数最小的cart树

$t_m={\arg\min}_{fm}L(Fm-1,y)+L'(Fm-1,y)*fm$

因此$t_m$的拟和值是目标函数对fm求偏导=0时的值，

$\frac{dc}{dfm}=L'(Fm-1,y)$

在GBDT分类中，这里的Fm-1=sigmoid(累加和)

因此这个损失函数的形式和求导过程如下(用x替换了Fm-1)：

$ylog(sigmoid(x))+(1-y)log(1-sigmoid(x))$

$\frac{y}{sigmoid(x)}*sigmoid(x)*(1-sigmoid(x)) + \frac{y-1}{1-sigmoid(x)}*sigmoid(x)*(1-sigmoid(x))$

$y-ysigmoid(x) + (y-1)(sigmoid(x))$

$y-ysigmoidx(x)+ysigmoid(x)-sigmoid(x)$

$y-sigmoid(x)$

所以，每轮拟合的“伪残差”是y-p=y-sigmoid(累加和)

实际求解可能用牛顿法，那就是拟合$\frac{y-p}{p(1-p)}$

