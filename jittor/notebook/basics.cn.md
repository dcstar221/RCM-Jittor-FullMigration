
# 基本概念：Op, Var 


要使用jittor训练模型，您需要了解两个主要概念：

* Var：Jittor的基本数据类型
* Operations：Jittor的算子与numpy类似


首先，让我们开始使用Var。Var是jittor的基本数据类型，为了运算更加高效Jittor中的计算过程是异步的。 如果要访问数据，可以使用`Var.data`进行同步数据访问。

```
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

 Jittor的算子与numpy类似。 让我们尝试一些操作， 我们通过操作jt.float32创建Var `a`和`b`，并将它们相加。 输出这些变量相关信息，可以看出它们具有相同的形状和类型。

```
import jittor as jt
a = jt.float32([1,2,3])
b = jt.float32([4,5,6])
c = a+b
print(a,b,c)
```


除此之外，我们使用的所有算子`jt.xxx(Var,...)`都具有别名`Var.xxx(...)`。 例如：

```
c.max() # alias of jt.max(a)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```


如果您想知道Jittor支持的所有操作，可以运行`help(jt.ops)`。 您在`jt.ops.xxx`中找到的所有操作都可以通过别名`jt.xxx`。

```
help(jt.ops)
```

