# Basics: Op, Var


To train your model with jittor, there are only two main concept you need to know:


* Var: basic data type of jittor
* Operations: Jittor'op is simular with numpy

## Var
First, let's get started with Var. Var is the basic data type of jittor. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.data` can be used for synchronous data accessing.


```
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```
## Op
Jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via operation `jt.float32`, and add them. Printing those variables shows they have the same shape and dtype.


```
import jittor as jt
a = jt.float32([1,2,3])
b = jt.float32([4,5,6])
c = a+b
print(a,b,c)
```

Beside that, All the operators we used `jt.xxx(Var, ...)` have alias `Var.xxx(...)`. For example:


```
c.max() # alias of jt.max(a)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

if you want to know all the operation which Jittor supports. try `help(jt.ops)`. All the operation you found in `jt.ops.xxx`, can be used via alias `jt.xxx`.


```
help(jt.ops)
```

