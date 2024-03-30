---
    title: Basics
    layout: ../../../layouts/blogpost.astro
---

# Basics

## Contents

## Introduction

JAX is an high performance computing library, designed with a Numpy-like API and with autograd built in. It follows the functional programming approach which means your functions must be pure (i.e. have no side-effects). Aside from the Numpy-esque API, JAX has a few very extremely powerful tricks up its sleeve:

- `jax.jit`
- `jax.vmap`
- `jax.grad`

## JIT

JAX allows you to easily JIT-compile your functions by using the `jax.jit(...)` function. Here's an example:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.mean(x)

f_jit = jax.jit(f)
f_jit(jnp.arange(10)) # function gets compiled here -- some delay
f_jit(jnp.arange(10)) # uses cached JITted function -- no delay
```

The function `f` will be compiled as soon as you call it and then will be cached for subsequent calls. The `jax.jit` function compiles your code through MLIR into XLA code.

This means that when you're using JAX, you aren't **really** writing Python code: you're writing XLA code! This distinction is very important, because it forces you to rethink about how to write code.

For example, let's think about the JIT compilation again. What exactly got compiled? Well, it's obviously the function `f` but more importantly, you have compiled a function which takes an array of size $10$ as input. If you changed the input, then JAX would re-compile the function, i.e.:

```python
import jax
import jax.numpy as jnp

def f(x):
    return jnp.mean(x)

f_jit = jax.jit(f)
f_jit(jnp.arange(10)) # function gets compiled here -- some delay
f_jit(jnp.arange(10)) # uses cached JITted function -- no delay
f_jit(jnp.arange(20)) # function gets compiled here again!
```

This means that effectly, you write functions for very specific array shapes -- there are no dynamic arrays in JAX. This is a stark contrast to Python where almost everything is dynamic.

As a rule of thumb: always try to JIT at the highest possible node. Try to get as many (JAX) computations as you can into a single JIT call.

## VMAP

In JAX, the function `jax.vmap` performs vectorisation across some axis. Often, this is used for the batch dimension sitting (typically) at `axis=0`. Whereas in PyTorch you would write the forward function in a way that handles the batch dimension, in JAX you typically don't bother with the batch dimension.

> PyTorch now also has a `vmap` [function](https://pytorch.org/docs/stable/generated/torch.vmap.html). I don't see it being used that often, but this will likely change in the near future.

For example, say you're training on MNIST and the data shape is $64\times 28\times 28$, where $64$ is the batch dimension. In JAX, you would write your function such that it accepts a $28 \times 28$ array and then use `jax.vmap` to take care of the batch dimension, i.e.:

```python
import jaxtyping as jt
import jax


def f(jt.Float[jt.Array, "width height"]):
    return some_computation()

data = get_mnist_batch() # 64 x 28 x 28

output = jax.jit(jax.vmap(f))(data) # the JIT is not necessary, but you typically shouldn't use vmap outside of JIT

```

As you can see, we defined the function `f` with the relevant shape ($28\times 28$) and mentally freed ourselves from the batch dimension. There's more you can do with the `vmap` function, for example you can define which axis to `vmap` over - per default it's the 0th axis. We will see more of this in later tutorials.

## GRAD

One of the most amazing features of JAX is the simple automatic differentiation, which is - in my opinion - the best API out there. Simply pass it a function as input and you get it's derivative as output. For example:

```python
import jax

def f(x):
    return x**2

f(8) # 64
f_prime = jax.grad(f)
f_prime(8) # 16

f_prime_prime = jax.grad(f_prime)
f_prime_prime(8) # 2
```

One thing to mention is that the `grad` function computes the gradient w.r.t. the first parameter of the function. If you have rewired your brain that you're writing XLA code (i.e. you're writing pure functions with well-defined shapes) then the `grad` function will work wonders. We will see more of this in later tutorials.

## Conclusion

JAX is an extremely powerful tool and we have barely scratched the surface. So stay tuned for more tutorials in the future.
