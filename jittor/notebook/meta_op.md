# Meta-operator: Implement your own convolution with Meta-operator


Meta-operator is a key concept of jittor, The hierarchical architecture of meta-operators is shown below.

The meta-operators are consist of reindex, reindex-reduce and element-wise operators. Reindex and reindex-reduce operators are both unary operators. The reindex operator is a one-to-many mapping between its input and output. And the reindex-reduce operator is a many-to-one mapping. Broadcast, pad and slice operators are common reindex operators. And reduce, product and sum are common reindex-reduce operators. Element-wise operator is the third component of meta-operators. Compared to the first two, element-wise operators may contain multiple inputs. But all the input and output shapes of element-wise operators must be the same. And they are one-to-one mapped. For example, the addition of two variables is a binary element-wise operator.



> ![](./figs/mop.svg)
> The hierarchical architecture of meta-operators. The meta-operators are consist of reindex, reindex-reduce and element-wise operators. Reindex and reindex-reduce are each other's backward operators. The backward operators of element-wise operators are itself. Those meta-operators are fused into common DL operations, and these DL operators further constitute the model.
>

In the previous [example](example.ipynb), we have demonstrated how to implement matrix multiplication via three meta-operators:


```
def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)
```

In this tutorial, we will show how to implement your own convolution with meta-operator.

First, let's implement a naive Python convolution:



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

Then, let's download a cat image, and run `conv_naive` with a simple horizontal filte.


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
It looks good, our `naive_conv` works well. Let's replace our naive implementation with jittor.


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

They looks the same. How about the performance?


```
%time y = conv_naive(x, w)
%time jy = conv(jx, jw).fetch_sync()
```

The jittor implementation is much faster. So why this two implementation are equivalent in math, and why jittor's implementation is faster? We will explain step by step:

First, let's take a look at the help document of `jt.reindex`.



```
help(jt.reindex)
```

Following the document, we can expand the reindex operation for better understanding:


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

**After expansion:**


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

**After loop fusion:**


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

This is the trick of meta-operator, It can fused multiple operator into a complicated operation, including many variation of convolution (e.g. group conv, seperate conv,...).

jittor will try to optimize the fused operator as fast as possible. Let's try some optimizations(compile the shapes as constants into the kernel), and show the underlying c++ kernel.




```
jt.flags.compile_options={"compile_shapes":1}
with jt.profile_scope() as report:
    jy = conv(jx, jw).fetch_sync()
jt.flags.compile_options={}

print(f"Time: {float(report[1][4])/1e6}ms")

with open(report[1][1], 'r') as f:
    print(f.read())
```

Even faster than the previous implementation! From the output we can look at the function definition of func0. This is the main code of our convolution kernel, which is generated Just-in-time. Because the compiler knows the shapes of the kernel and more optimizations are used. 



In this tutorial, Jittor simply demonstrated the use of meta-operators, which is not a performance test. If you need a performance test, `jt.flags.enable_tuner = 1` will try to use the dedicated hardware library.
