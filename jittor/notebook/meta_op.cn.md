
# 元算子：通过元算子实现自己的卷积层



元算子是jittor的关键概念，元算子的层次结构如下所示。

元算子由重索引算子，重索引化简算子和元素级算子组成。重索引算子，重索引化简算子都是一元算子。 重索引算子是其输入和输出之间的一对多映射。重索引简化算子是多对一映射。广播，填补, 切分算子是常见的重新索引算子。 而化简，累乘，累加算子是常见的索引化简算子。 元素级算子是元算子的第三部分，与前两个相比，元素算级子可能包含多个输入。 但是元素级算子的所有输入和输出形状必须相同，它们是一对一映射的。 例如，两个变量的加法是一个二进制的逐元素算子。

>
> 元算子的层级结构。元算子包含三类算子,重索引算子,重索引化简算子,元素级算子。元算
> 子的反向传播算子还是元算子。元算子可以组成常用的深度学习算子。而这些深度学习算子又
> 可以进一步组成深度学习模型。


在第一个[示例](example.ipynb)中，我们演示了如何通过三个元算子实现矩阵乘法：

```
def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)
```



在本教程中，我们将展示如何使用元算子实现自己的卷积。

首先，让我们实现一个朴素的Python卷积：

```
import numpy as np
import os
def conv_naive(x, w):
    N,H,W,C = x.shape

    Kh, Kw, _C, Kc = w.shape
    assert C==_C, (x.shape, w.shape)
    y = np.zeros([N,H-Kh+1,W-Kw+1,Kc])
    for i0 in range(N):
        for i1 in range(H-Kh+1): # dimension error
            for i2 in range(W-Kw+1):
                for i3 in range(Kh):
                    for i4 in range(Kw):
                        for i5 in range(C):
                            for i6 in range(Kc):
                                if i1-i3<0 or i2-i4<0 or i1-i3>=H or i2-i4>=W: continue
                                y[i0, i1, i2, i6] += x[i0, i1 + i3, i2 + i4, i5] * w[i3,i4,i5,i6]
    return y
```


然后，让我们下载一个猫的图像，并使用`conv_naive`实现一个简单的水平滤波器。

```
# %matplotlib inline
import pylab as pl
img_path="/tmp/cat.jpg"
if not os.path.isfile(img_path):
    !wget -O - 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Felis_silvestris_catus_lying_on_rice_straw.jpg/220px-Felis_silvestris_catus_lying_on_rice_straw.jpg' > $img_path
img = pl.imread(img_path)
pl.subplot(121)
pl.imshow(img)
kernel = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1],
])
pl.subplot(122)
x = img[np.newaxis,:,:,:1].astype("float32")
w = kernel[:,:,np.newaxis,np.newaxis].astype("float32")
y = conv_naive(x, w)
print (x.shape, y.shape) # shape exists confusion
pl.imshow(y[0,:,:,0])
```

看起来不错，我们的`naive_conv`运作良好。现在让我们用jittor替换我们的朴素实现。

```
import jittor as jt

def conv(x, w):
    N,H,W,C = x.shape
    Kh, Kw, _C, Kc = w.shape
    assert C==_C
    xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
        'i0', # Nid
        'i1+i3', # Hid+Khid
        'i2+i4', # Wid+KWid
        'i5', # Cid|
    ])
    ww = w.broadcast_var(xx)
    yy = xx*ww
    y = yy.sum([3,4,5]) # Kh, Kw, c
    return y

# Let's disable tuner. This will cause jittor not to use mkl for convolution
jt.flags.enable_tuner = 0

jx = jt.array(x)
jw = jt.array(w)
jy = conv(jx, jw).fetch_sync()
print (jx.shape, jy.shape)
pl.imshow(jy[0,:,:,0])
```


他们的结果看起来一样。那么它们的性能如何？

```
%time y = conv_naive(x, w)
%time jy = conv(jx, jw).fetch_sync()
```



可以看出jittor的实现要快得多。 那么，为什么这两个实现在数学上等效，而jittor的实现运行速度更快？ 我们将逐步进行解释：

首先，让我们看一下`jt.reindex`的帮助文档。

```
help(jt.reindex)
```


遵循该文档，我们可以扩展重索引操作以便更好地理解：

```
py
xx = x.reindex([N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc], [
    'i0', # Nid
    'i1+i3', # Hid+Khid
    'i2+i4', # Wid+KWid
    'i5', # Cid
])
ww = w.broadcast_var(xx)
yy = xx*ww
y = yy.sum([3,4,5]) # Kh, Kw, C
```


扩展后：

```
py
shape = [N,H-Kh+1,W-Kw+1,Kh,Kw,C,Kc]
# expansion of x.reindex
xx = np.zeros(shape, x.dtype)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                for i4 in range(shape[4]):
                    for i5 in range(shape[5]):
                        for i6 in range(shape[6]):
                            if is_overflow(i0,i1,i2,i3,i4,i5,i6):
                                xx[i0,i1,...,in] = 0
                            else:
                                xx[i0,i1,i2,i3,i4,i5,i6] = x[i0,i1+i3,i2+i4,i5]

# expansion of w.broadcast_var(xx)
ww = np.zeros(shape, x.dtype)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                for i4 in range(shape[4]):
                    for i5 in range(shape[5]):
                        for i6 in range(shape[6]):
                            ww[i0,i1,i2,i3,i4,i5,i6] = w[i3,i4,i5,i6]
# expansion of xx*ww
yy = np.zeros(shape, x.dtype)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                for i4 in range(shape[4]):
                    for i5 in range(shape[5]):
                        for i6 in range(shape[6]):
                            yy[i0,i1,i2,i3,i4,i5,i6] = xx[i0,i1,i2,i3,i4,i5,i6] * ww[i0,i1,i2,i3,i4,i5,i6]
# expansion of yy.sum([3,4,5])
shape2 = [N,H-Kh+1,W-Kw+1,Kc]
y = np.zeros(shape2, x.dtype)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                for i4 in range(shape[4]):
                    for i5 in range(shape[5]):
                        for i6 in range(shape[6]):
                            y[i0,i1,i2,i6] += yy[i0,i1,i2,i3,i4,i5,i6]
```


循环融合后：

```
py
shape2 = [N,H-Kh+1,W-Kw+1,Kc]
y = np.zeros(shape2, x.dtype)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                for i4 in range(shape[4]):
                    for i5 in range(shape[5]):
                        for i6 in range(shape[6]):
                            if not is_overflow(i0,i1,i2,i3,i4,i5,i6):
                                y[i0,i1,i2,i6] += x[i0,i1+i3,i2+i4,i5] * w[i3,i4,i5,i6]
```



这是就元算子的优化技巧，它可以将多个算子融合为一个复杂的融合算子，包括许多卷积的变化（例如group conv，separate conv等）。

jittor会尝试将融合算子优化得尽可能快。 让我们尝试一些优化（将形状作为常量编译到内核中），并编译到底层的c++内核代码中。


```
jt.flags.compile_options={"compile_shapes":1}
with jt.profile_scope() as report:
    jy = conv(jx, jw).fetch_sync()
jt.flags.compile_options={}

print(f"Time: {float(report[1][4])/1e6}ms")

with open(report[1][1], 'r') as f:
    print(f.read())
```


比之前的实现还要更快！ 从输出中我们可以看一看`func0`的函数定义，这是我们卷积内核的主要代码，该内核代码是即时生成的。因为编译器知道内核的形状，所以使用了更多的优化方法。

在这个教程中，Jittor简单演示了元算子的使用，并不是正真的性能测试，所以使用了比较小的数据规模进行测试，如果需要性能测试，请打开`jt.flags.enable_tuner = 1`，会启动使用专门的硬件库加速。

