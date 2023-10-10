<script lang="ts">
    import Figure from "$lib/components/Figure.svelte";
    import Katex from "$lib/components/Katex.svelte";
</script>

<h3>Introduction</h3>
<p>
    Previously, we had implemented a regular neural network, but this time we
    will step it up a notch by creating a convolutional neural network. But
    before we begin, I'm going to set a prerequisite for this blogpost. I want
    you to watch 3B1B's video on convolutions first before you continue with
    this video on convolutions. Specifically, I want you to know at least what
    the computation of a cross-correlation and convolution looks like and also
    you should know what the difference is between those two operations. If you
    have at least a rough idea, you can continue. If not, you can still
    continue, but at some moments it might not click for you as much as it would
    have, if you watched 3B1B's video first. Before we get into the nitty-gritty
    details of a convolutional layer, we first have to look at our dataset and
    in the previous episodes, I had used the MNIST dataset, but to make the
    derivations of the math simpler, I'm going to create a custom dataset.
</p>
<h3>A Simple Dataset</h3>
<p>
    In general, our dataset will be a numpy tensor with this shape <Katex
        math={"x: [n \\times c_{in} \\times h \\times w]"}
    /> where <Katex math={"n"} /> is the batch size, <Katex math={"c_{in}"} /> is
    the number of input channels (we'll get to those later) and <Katex
        math={"H"}
    /> and <Katex math={"W"} /> are the height and the width of the input image.
    The input channel (at least in the case of image processing) often refers to
    the number of colour channels. For example, a grey-scaled image has only a single
    channel, which gives us the darkness of any pixel. A colour image would have
    3 channels, one for the red, green and blue value of a pixel. In our simple dataset,
    we're only going to have a single channel.
</p>
<div class="flex space-x-4">
    <Figure path="CNN Numpy.avif" caption="Grey-scaled image" />
    <Figure path="CNN Numpy Colour.avif" caption="Colour image" />
</div>
<p>
    Our networks task will be to detect shapes. Here, I've predefined a couple
    of blueprints for the shapes and their corresponding target values.
</p>
