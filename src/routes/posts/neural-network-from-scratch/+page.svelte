<script lang="ts">
    import Katex from "$lib/components/Katex.svelte";
</script>

<p>
    A neural network (NN) is a computational model, which took inspiration from
    biological brains. NNs can be treated as universal function approximators,
    meaning that - given enough training data - a NN can learn any function (in
    theory). In this blog post, we will implement a NN using nothing but Numpy.
    But first, we need to discuss the basics.
</p>

<p>
    A NN is structured in 3 parts: the input layer, some hidden layers and the
    output layer. We represent these layers (and the weights and biases that
    connect them) using matrix notation. For example, if your training data
    consists of <Katex math={"m=25"} /> examples with each having <Katex
        math={"n=2"}
    /> features, the resulting matrix for your input layer would have the shape
    <Katex math={"X:[m\\times n],"} displayMode /> where
    <Katex math={"m"} /> is the number of training examples and <Katex
        math={"n"}
    /> is the number of features. A feature in your training data is just some characteristic
    that describes your data. For example, if you wanted to predict housing prices,
    possible features could be the age of the house or the living area in <Katex
        math={"m^2"}
    />.
</p>

<p>
    Let's say our NN has only 1 hidden layer with 16 neurons. We denote the
    number of neurons in the hidden layer as <Katex math={"h_1"} />
    . Since we have <Katex math={"m"} /> training examples, we also need to have
    <Katex math={"m"} /> rows in our hidden layer - one for each training example.
    This means that our hidden layer <Katex math={"H_1"} /> has the shape of
    <Katex math={"H_1: [m \\times h_1]"} displayMode />
</p>
