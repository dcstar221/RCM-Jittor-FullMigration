
# 示例：模型定义与训练


以下示例展示了如何逐步搭建两层神经网络模型，并通过几行Python代码从头开始进行模型训练。

```
import jittor as jt
import numpy as np
from jittor import nn, Module, init
```


以下代码定义了我们的模型，该模型是一个两层神经网络。 隐藏层的大小为10，激活函数为relu。

```
### model define

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```


最后，从头开始训练该模型。 优化器使用简单的梯度下降，损失函数为L2距离。 为提高效率训练过程是异步的。 jittor通过**统一计算图**和**即时分析器**计算梯度，并进行计算图级和算子级的优化。

在该示例中，Jittor使用了多个优化，包括：**算子融合**，激活函数和损失函数可以融合到第一和第二全连接层中； 矩阵乘法中的三元算子也可以融合。 **并行化**，它可以提高现代多核CPU和GPU上计算密集型运算的性能。 算子融合是一种计算图级优化，而并行化则同时作用于图形级和算子级的优化。

```
np.random.seed(0)
jt.set_seed(3)
n = 1000
batch_size = 50
base_lr = 0.05
# we need to stop grad of global value to prevent memory leak
lr = jt.float32(base_lr).name("lr").stop_grad()

def get_data(n):
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.1
optim = nn.SGD (model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    loss = jt.sqr(pred_y - y)
    loss_mean = loss.mean()
    optim.step (loss_mean)
    print(f"step {i}, loss = {loss_mean.data.sum()}")

assert loss_mean.data < 0.005
```
