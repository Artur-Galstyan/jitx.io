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

def f(x):
    return x ** 2

f_jit = jax.jit(f)
print(f_jit(2)) # function gets compiled here -- some delay
print(f_jit(2)) # uses cached JITted function -- no delay
```

The function `f` will be compiled as soon as you call it and then will be cached for subsequent calls. The `jax.jit` function compiles your code through MLIR into XLA code.

This means that when you're using JAX, you aren't **really** writing Python code: you're writing XLA code! This distinction is very important, because it forces you to rethink about how to write code.
