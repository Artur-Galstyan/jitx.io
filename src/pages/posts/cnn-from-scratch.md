---
    layout: ../../layouts/blogpost.astro
    title: Convolutional Neural Network from Scratch
    pubDate: 2023-01-31 
    description: "Implementing a convolutional neural network with our bare hands (and Numpy)"
    tags: ["cnn", "numpy"]
---

# Convolutional Neural Network from Scratch

2023-01-31

## Contents

## Introduction

Previously, we had implemented a regular neural network, but this time we will step it up a notch by creating a convolutional neural network. But before we begin, I'm going to set a prerequisite for this blogpost. I want you to watch 3B1B's video on convolutions first before you continue with this video on convolutions. Specifically, I want you to know at least what the computation of a cross-correlation and convolution looks like and also you should know what the difference is between those two operations. If you have at least a rough idea, you can continue. If not, you can still continue, but at some moments it might not click for you as much as it would have, if you watched 3B1B's video first. Before we get into the nitty-gritty details of a convolutional layer, we first have to look at our dataset and in the previous episodes, I had used the MNIST dataset, but to make the derivations of the math simpler, I'm going to create a custom dataset.

## A Simple Dataset

In general, our dataset will be a numpy tensor with this shape $x: [n \times c_{in} \times h \times w]$ where $n$ is the batch size, $c_{in}$ is
the number of input channels (we'll get to those later) and
$H$ and $W$ are the height and the width of the input image.
The input channel (at least in the case of image processing) often refers to
the number of colour channels. For example, a grey-scaled image has only a single
channel, which gives us the darkness of any pixel. A colour image would have
3 channels, one for the red, green and blue value of a pixel. In our simple dataset,
we're only going to have a single channel.

<div class="figures">
    <img class="half" src="/posts/cnn-from-scratch/CNN_Numpy.avif" />
    <img class="half" src="/posts/cnn-from-scratch/CNN_Numpy_Colour.avif" />
</div>

Our networks task will be to detect shapes. Here, I've predefined a couple of blueprints for the shapes and their corresponding target values.

```python
V_LINE = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]).reshape(C_IN, H, W)
H_LINE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]).reshape(C_IN, H, W)
CROSS = np.clip(V_LINE + H_LINE, a_min=0, a_max=1).reshape(C_IN, H, W)
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).reshape(C_IN, H, W)

_, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4) # 1 row, 4 columns


ax1.imshow(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]))
ax2.imshow(np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
ax3.imshow(np.clip(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) + np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]), a_min=0, a_max=1))
ax4.imshow(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))

plt.tight_layout()


SHAPES = {
    "v_line":V_LINE,
    "h_line": H_LINE,
    "cross": CROSS,
    "diamond": DIAMOND
}
TARGETS = {
    "v_line":0,
    "h_line": 1,
    "cross": 2,
    "diamond": 3
}
```

![Custom Shapes](/posts/cnn-from-scratch/customshapes.png)

Our dataset will consist of these, but not exactly these, because our network would easily just memorise these values. Instead, we will use these as a blueprint and add a bit of random noise on top to make sure that our network doesn't just simply memorise these numbers.

```python
def generate_single_data_point(shape: str):
    return TARGETS[shape], (SHAPES[shape] + np.random.uniform(-0.25, 0.25, size=(C_IN, H, W)))

def generate_dataset(n=500, batch_size=16):
    def split_into_batches(x, batch_size):
        n_batches = len(x) / batch_size
        x = np.array_split(x, n_batches)
        return np.array(x, dtype=object)
    one_hot_encoder = OneHotEncoder()

    shapes = list(SHAPES.keys())
    dataset = [generate_single_data_point(np.random.choice(shapes)) for _ in range(n)]
    labels = []
    data = []
    for y, x in dataset:
        labels.append(y)
        data.append(x)

    targets = np.array(labels)
    data = np.array(data)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.33, random_state=RANDOM_SEED)

    X_train = split_into_batches(
        X_train, batch_size
    )

    X_test = split_into_batches(
        X_test, batch_size
    )

    # Turn the targets into Numpy arrays and flatten the array
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    # One-Hot encode the training data and split it into batches (same as with the training data)
    one_hot_encoder.fit(y_train)
    y_train = one_hot_encoder.transform(y_train).toarray()
    y_train = split_into_batches(np.array(y_train), batch_size)

    one_hot_encoder.fit(y_test)
    y_test = one_hot_encoder.transform(y_test).toarray()
    y_test = split_into_batches(np.array(y_test), batch_size)

    return X_train, y_train, X_test, y_test
```

This function `generate_dataset(...)` basically generates 500 datapoints with their corresponding, one-hot encoded target values and then splits the data into a training and testing set. I won't go into detail what this function does, because the exact procedure of this function doesn't really matter to us right now. Also you don't need any machine learning knowledge to understand this code, so if you want to, you can look at this later on your own free time. For now, we'll treat this as a blackbox that simply generates the dataset for us.

Now that we have the data, we can finally start with forward propagation.

## Forward Propagation

As mentioned before, on this left side, we have our input data with the
familiar shape. Note that each of these boxes is a whole matrix. In the
middle, we have the heart of our convolutional layer, namely the kernels and
the biases, which represent the trainable parameters of the convolutional
layer. We call each column here a kernel, and each kernel consists of a set
of matrices and a single bias value. Each kernel has as many matrices as we
have input channels and we will see why that is the case in just a moment.
The number of kernels defines the shape of the output of the convolutional
layer and that number is called $c*{out}$ or the number of output
channels. And since each kernel matrix is a squared matrix (which is doesn't
have to be, I just like it this way), the final shape of the kernels is
$[c*{out} \times c_{in} \times t \times t]$ . We also
have the bias matrix, which is just a single row matrix. We use this comma
notation, to tell numpy to perform broadcasting and I've covered
broadcasting in my previous blogposts.

![Forward Propagation](/posts/cnn-from-scratch/ForwardPropagation1.avif)

The output of the convolutional layer is another set of matrices. While the
input had $c*{in}$ as the number of channels, for our output,
the number of channels is $c*{out}$ . Also, each box here is
a matrix, with the shape $p \times q$ />, and these values
depend on the width and height of our input matrices as well as our kernels.
More specifically, what you see here is the resulting shape of a so-called
valid cross-correlation between two matrices.

For the actual forward propagation, we will take each of the input channel
matrices and compute the cross-correlation between them and the matrices of
the first kernel. Now you should see why each kernel must have exactly $c_{in}$ matrices, because each channel of the input needs to be cross-correlated with
a kernel matrix. Then, we take the sum of each of these cross-correlations and
add the bias on top. Remember, that the bias is a single number, but that's fine,
since numpy will perform broadcasting for us and that single number will be added
to every value of the matrix. The result of all of this is the first of the output
matrices. We then repeat this process for every kernel.

<div class="figures">
    <img class="half" src="/posts/cnn-from-scratch/fp2.avif" />
    <img class="half" src="/posts/cnn-from-scratch/fp1.avif" />
</div>

More generally, we can write the formula for the forward propagation like this. This is where we have to bring back our batch size, that we omitted in the beginning. This forward propagation step has to be performed for every training example in the batch, which means we have to iterate over our batch and then perform forward propagation for every datapoint.

![Forward Propagation](/posts/cnn-from-scratch/fpeq.avif)

Let's implement this in Python.

```python
def conv2d_forward(x: np.array, kernels: np.array, bias: np.array):
    c_out, c_in, k, _ = kernels.shape
    N, c_in, h_in, w_in = x.shape
    h_out = h_in - k + 1
    w_out = w_in - k + 1
    output_shape = N, c_out, h_out, w_out
    output = np.zeros(shape=output_shape)
    for n in range(N):
        for j in range(c_out):
            output[n, j] = bias[j]
            for i in range(c_in):
                output[n, j] += correlate2d(
                    x[n, i], kernels[j, i], mode="valid"
                )
    return output
```

First we gather all the hyper parameters from our input. We can get the
number of channels from the kernels and the batch size and the dimensions of
the image from the input matrix. With that we initialise the empty output
array. We then iterate over every sample in our batch and perform forward
propagation pretty much exactly as we've written it in our formula. This
`correlate2d()` function comes from SciPy and is a neat helper for
us. In Numpy, there is only the “normal” cross-correlation which - in mathematical
terms - is an operation between two functions, not necessarily between matrices.
This function from ScipPy does this operation on 2d matrices, so we won't have
to implement that from scratch. Finally, we return the output.

This was it for the forward propagation step, but the difficult part is yet
to come, which is the back propagation step in which we update the trainable
parameters of our layers.

## Backpropagation Overview

As always, we get as input the error gradient, which can either come from
this layer being the output layer of the whole network, or our layer being
just some layer in the middle, meaning we get it from the layer in front.

![Backpropagation Overview](/posts/cnn-from-scratch/bpoverview.avif)

Either way, we have to use this error gradient to compute the partial
derivatives for the kernel and the bias. Then, the output of the back
propagation is the input error, which we pass to the layers before us. That
input error then becomes the error gradient for the layer behind us. Then
that layer computes the gradients and passes the error backwards and so on.
So to summarise, we get the error gradient matrix as input and we have to
compute the gradient matrices: the kernel gradient, the bias gradient and
the input gradient. Let's start with the kernel gradient.

## Kernel Gradient

So far this is all the information we have:

![Kernel Gradient](/posts/cnn-from-scratch/k1.avif)

If we write out the forward propagation equation, we get this for $j=0$, or in other words, the first set of kernels.

![Kernel Gradient](/posts/cnn-from-scratch/k2.avif)

Our goal is to find a generic formula such that we can compute any arbitrary
kernel gradient, for example this kernel gradient for $j=0$
and $i=1$. Since $j=0$, we look at the
equation of
$y$ where $j=0$. And although all of these
terms here are matrices and we can't use the chain rule directly, we can
still use the general idea of the chain rule by asking ourselves, how much
influence did $K_0^1$ have on the error gradient.

![Kernel Gradient](/posts/cnn-from-scratch/k3.avif)

Well, we can see in this equation, that $K_0^1$ only appears
once here. And, since we have a sum here, when we would take the derivative of
this output, everything else would become $0$ anyway, except
the part where our kernel had some influence. This means that the only relevant
part of this equation, as far as $K_0^1$ is concerned, can be
boiled down to this part:

![Kernel Gradient](/posts/cnn-from-scratch/k4.avif)

In our concrete example, with our parameters, this can be written like so in
matrix form.

![Kernel Gradient](/posts/cnn-from-scratch/k5.avif)

We can expand the equations of the cross-correlation, giving us this set of
equations.

![Kernel Gradient](/posts/cnn-from-scratch/k6.avif)

And, just to remind ourselves, our goal is to compute the kernel gradient
given the output gradient.

![Kernel Gradient](/posts/cnn-from-scratch/k7.avif)

For this, we can use the chain rule. I know, this looks confusing, but once
you break it down, it really isn't.

![Kernel Gradient](/posts/cnn-from-scratch/k8.avif)

For example, in this first equation, we are looking for all the
contributions of $K*{00}^{01}$ towards the error gradient. We
notice, that
$K*{00}^{01}$ appears in all 4 equations, meaning we have 4
contributions of
$K_{00}^{01}$
and we have to add those up. The same goes for all the other values of the kernel.
Lucky for us, these gradients are very easy to calculate and all we're left with
is some part of our input matrix $X$ .

![Kernel Gradient](/posts/cnn-from-scratch/k9.avif)

It looks as though we have some sort of operation between our input matrix
and the error gradient.

![Kernel Gradient](/posts/cnn-from-scratch/k10.avif)

What we have here is the valid cross-correlation that we had previously used
in the forward propagation step. At this point, it might help to take a
moment to verify that this really is the cross-correlation operation.Now, we
can bring this back to matrix form and taking this one step further, we can
write the generic formula to compute any kernel gradient we want.

![Kernel Gradient](/posts/cnn-from-scratch/k11.avif)

We can now take this formula and implement this in Python.

```python
def conv2d_kernel_gradient(dE_dY: np.array, x: np.array):
    n, c_out, h_out, w_out = dE_dY.shape
    _, c_in, h_in, w_in = x.shape
    k = h_in + 1 - h_out
    dK = np.zeros(shape=(c_out, c_in, k, k))
    for batch in range(n):
        for i in range(c_out):
            for j in range(c_in):
                dK[i, j] += correlate2d(dE_dY[batch, i], x[batch, j], mode="valid")

    return dK / n
```

We once again first take all the necessary parameters from our inputs and we
also reverse engineer what the shape of the kernels is. Then, we iterate
over all our batches and compute the gradients exactly as in the formula.
One extra step we have to take care of is that we're adding up all the
gradients from every training sample. In the end, we divide the kernel
gradient by the batch size, which gives us the average kernel gradient. If
we didn't do this part, we would have one giant kernel gradient, but instead
we want to have the average kernel gradient, which is why we divide this by
the batch size in the end.

his was the kernel gradient. Next up is the bias gradient.

## Bias Gradient
