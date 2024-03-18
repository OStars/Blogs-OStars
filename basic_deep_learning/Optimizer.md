# 优化器 / 优化算法

深度学习优化算法经历了 SGD -> SGDM -> NAG ->AdaGrad -> AdaDelta -> Adam -> Nadam -> AdamW 这样的发展历程，但是他们都可以用以下这个统一的框架来描述：

**定义：**待优化参数 $w$，目标函数 $f(w)$，初始学习率 $\alpha$，当前迭代 epoch 为 $t$

**步骤：**

1. 计算目标函数关于当前参数的梯度：
   $$
   g_t = \nabla f(w_t)
   $$

2. 根据**历史梯度**计算一阶动量和二阶动量：
   $$
   m_t = \phi(g_1,g_2,...,g_{t-1});\quad V_t=\psi(g_1,g_2,...,g_{t-1})
   $$

3. 计算当前时刻的下降梯度：
   $$
   \eta_t = \alpha*m_{t-1}/\sqrt{V_{t-1}}
   $$

4. 根据下降梯度进行参数更新：
   $$
   w_{t+1} = w_t - \eta_t
   $$
   (不过这里这个梯度 $\eta_t$ 前面可能还有会一个系数，因为可能会有 lr schedule 的存在)

无论是哪一种优化算法，最后两个步骤都是一样的 (有的没有一阶动量那就 $m_{t-1}=g_t$，有的没有二阶动量那就 $V_{t-1}=I^2$，其中 $I^2$ 表示单位矩阵的二阶矩)，主要区别在于前两步，尤其是计算一阶动量和二阶动量的方法不同

同时，在所有优化器的代码里面有一些函数也是相通的：

* `add_param_group`(param_group)：把参数放进优化器中，这在 Fine-tune 预训练网络时很有用，因为可以使冻结层可训练并随着训练的进行添加到优化器中。
* `load_state_dict`(state_dict)：把优化器的状态加载进去。
* `state_dict`()：返回优化器的状态，以dict的形式返回。
* `step`(closure=None)：优化一步参数。
* `zero_grad`(set_to_none=False)：把所有的梯度值设为0。

## SGD

随机梯度下降，完全没有动量的概念， $m_{t-1}=g_t$，$V_{t-1}=I^2$

SGD 最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点。

## SGD with Momentum

为了抑制 SGD 的震荡，SGDM 认为梯度下降过程可以加入惯性。下坡的时候，如果发现是陡坡，那就利用惯性跑的快一些。SGDM 全称是 SGD with momentum，在SGD基础上引入了一阶动量：
$$
m_t = \beta_1*m_{t-1} + (1 - \beta_1)*g_t
$$
一阶动量是**各个时刻梯度方向的指数平均值**，约等于最近 $1/(1−\beta_1)$​ 个时刻的梯度向量和的平均值。

也就是说， $t$ 时刻的下降方向，不仅由当前点的梯度方向决定，而且由此前累积的下降方向决定。 $\beta_1$​ 的经验值为 0.9，这就意味着下降方向主要是此前累积的下降方向，并略微偏向当前时刻的下降方向。想象高速公路上汽车转弯，在高速向前的同时略微偏向，急转弯可是要出事的。

## SGD with Nesterov Acceleration

SGD 还有一个问题是困在局部最优的沟壑里面震荡。想象一下你走到一个盆地，四周都是略高的小山，你觉得没有下坡的方向，那就只能待在这里了。可是如果你爬上高地，就会发现外面的世界还很广阔。因此，我们不能停留在当前位置去观察未来的方向，而要向前一步、多看一步、看远一些。

NAG 全称 Nesterov Accelerated Gradient，是在 SGD、SGD-M 的基础上的进一步改进，改进点在于步骤 1。我们知道在时刻 $t$​ 的主要下降方向是由累积动量决定的，自己的梯度方向说了也不算，**那与其看当前梯度方向，不如先看看如果跟着累积动量走了一步，那个时候再怎么走**。因此，**NAG 在步骤 1，不计算当前位置的梯度方向，而是计算如果按照累积动量走了一步，那个时候的下降方向：**
$$
g_t=\nabla f(w_t-\beta_1*m_{t-1}/\sqrt{V_{t-1}})
$$
然后用下一个点的梯度方向，与历史累积动量相结合，计算步骤2中当前时刻的累积动量。(就是 SGDM 中 $m_t$ 中的  $g_t$ 替换成上面的表达式，说白了 SGD+NAG 就基本只看累计动量，基本不看当前梯度)

## AdaGrad

此前我们都没有用到二阶动量。二阶动量的出现，才意味着“自适应学习率”优化算法时代的到来。SGD 及其变种以同样的学习率更新每个参数，但深度神经网络往往包含大量的参数，这些参数并不是总会用得到（想想大规模的embedding）。对于经常更新的参数，我们已经积累了大量关于它的知识，不希望被单个样本影响太大，希望学习速率慢一些；对于偶尔更新的参数，我们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，即学习速率大一些。

怎么样去度量历史更新频率呢？那就是二阶动量——该维度上，迄今为止所有梯度值的平方和：
$$
V_t=\sum_{\tau=1}^t{g_\tau^2}
$$
再看计算下降梯度的公式：
$$
\eta_t = \alpha*m_{t-1}/\sqrt{V_{t-1}}
$$
可以看出，此时实质上的学习率由 $\alpha$ 变成了 $\alpha/\sqrt{V_{t-1}}$。 一般为了避免分母为 0，会在分母上加一个小的平滑项 (默认为 1e-10)。因此 $\sqrt{V_{t-1}}$​ 是恒大于 0 的，而且参数更新越频繁，二阶动量越大，学习率就越小。

**这一方法在稀疏数据场景下表现非常好。但也存在一些问题：因为 $\sqrt{V_{t-1}}$ 是单调递增的，会使得学习率单调递减至 0，可能会使得训练过程提前结束，即便后续还有数据也无法学到必要的知识。**

## AdaDelta / RMSProp

由于 AdaGrad 单调递减的学习率变化过于激进，我们考虑一个改变二阶动量计算方法的策略：不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。这也就是 AdaDelta 名称中 Delta 的来历。

修改的思路很简单。前面我们讲到，指数移动平均值大约就是过去一段时间的平均值，因此我们用这一方法来计算二阶累积动量：(就像计算一阶动量一样来计算二阶动量)
$$
V_t=\beta_2*V_{t-1}+(1-\beta_2)g_t^2
$$
然后计算下降梯度：
$$
\eta_t = \alpha/\sqrt{V_{t-1}}
$$
这就避免了二阶动量持续累积、导致训练过程提前结束的问题了。

## Adam

谈到这里，Adam 和 Nadam 的出现就很自然而然了——它们是前述方法的集大成者。我们看到，SGD-M 在SGD 基础上增加了一阶动量，AdaGrad 和 AdaDelta 在 SGD 基础上增加了二阶动量。把一阶动量和二阶动量都用起来，就是Adam 了——Adaptive + Momentum。

SGD的一阶动量：
$$
m_t = \beta_1*m_{t-1} + (1 - \beta_1)*g_t
$$
加上 AdaDelta 的二阶动量：
$$
V_t=\beta_2*V_{t-1}+(1-\beta_2)g_t^2
$$
论文说在迭代初始阶段，$m_t$ 和 $v_t$​ 会向初值偏移，所以还要对其进行矫正
$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t}	\\
\hat{V}_t=\frac{V_t}{1-\beta_2^t}
$$
优化算法里最常见的两个超参数 $\beta1$ 和 $\beta_2$ 就都在这里了，前者控制一阶动量，后者控制二阶动量。参数的默认值为： $\beta_1=0.9$， $\beta_2=0.999$，也就是说，一阶动量关注的时间窗口更小，大约有 10 个时间步；二阶动量的时间窗口则有 1000。

## Nadam

最后是 Nadam，我们说 Adam 是集大成者，但它居然遗漏了 Nesterov，按照NAG的步骤1：
$$
g_t=\nabla f(w_t-\beta_1*m_{t-1}/\sqrt{V_{t-1}})
$$
这就是 Nesterov + Adam = Nadam 了。

## AdamW

简单来说，**AdamW 就是 Adam 优化器加上 L2 正则，来限制参数值不能太大。**以往的 L2 正则是直接加在损失函数上的，但是 AdamW 选择加在了计算的梯度后，也就是
$$
\eta_t = \alpha*m_{t-1}/\sqrt{V_{t-1}} + \lambda\theta_{t-1}
$$
为什么这么做，BERT 原文里有如下描述：

> Just adding the square of the weights to the loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact with the m and v parameters in strange ways. Instead we want to decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent to adding the square of the weights to the loss with plain (non-momentum) SGD. Add weight decay at the end (fixed version).

这段话意思是说，如果直接将 L2 正则加到 loss 上去，由于 Adam 优化器的后序操作，该正则项将会与 $m_t$ 和 $V_t$ 产生奇怪的作用。因而，AdamW选择将 L2 正则项加在了 Adam 的 $m_t$ 和 $V_t$ 等参数被计算完之后，计算梯度时假如 L2 正则，所以这也表明了 weight_decay 和 L2 正则虽目的一致、公式一致，但用法还是不同，二者有着明显的差别。