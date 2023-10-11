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
<div class="flex space-x-4 justify-center">
    <Figure path="CNN Numpy.avif" caption="Grey-scaled image" />
    <Figure path="CNN Numpy Colour.avif" caption="Colour image" />
</div>
<p>
    Our networks task will be to detect shapes. Here, I've predefined a couple
    of blueprints for the shapes and their corresponding target values.
</p>
<pre><code class="language-python"
        >{`
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
`}</code
    ></pre>
<Figure path="customshapes.png" caption="Our Custom Shapes" />
<p>
    Our dataset will consist of these, but not exactly these, because our
    network would easily just memorise these values. Instead, we will use these
    as a blueprint and add a bit of random noise on top to make sure that our
    network doesn't just simply memorise these numbers.
</p>
<pre><code class="language-python"
        >{`
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

    return X_train, y_train, X_test, y_test`}</code
    ></pre>
<p>
    This function <code>generate_dataset(...)</code> basically generates 500 datapoints
    with their corresponding, one-hot encoded target values and then splits the data
    into a training and testing set. I won't go into detail what this function does,
    because the exact procedure of this function doesn't really matter to us right
    now. Also you don't need any machine learning knowledge to understand this code,
    so if you want to, you can look at this later on your own free time. For now,
    we'll treat this as a blackbox that simply generates the dataset for us.
</p>
<p>Now that we have the data, we can finally start with forward propagation.</p>
<h3>Forward Propagation</h3>
<p>
    As mentioned before, on this left side, we have our input data with the
    familiar shape. Note that each of these boxes is a whole matrix. In the
    middle, we have the heart of our convolutional layer, namely the kernels and
    the biases, which represent the trainable parameters of the convolutional
    layer. We call each column here a kernel, and each kernel consists of a set
    of matrices and a single bias value. Each kernel has as many matrices as we
    have input channels and we will see why that is the case in just a moment.
    The number of kernels defines the shape of the output of the convolutional
    layer and that number is called <Katex math={"c_{out}"} /> or the number of output
    channels. And since each kernel matrix is a squared matrix (which is doesn't
    have to be, I just like it this way), the final shape of the kernels is
    <Katex math={"[c_{out} \\times c_{in} \\times t \\times t]"} />. We also
    have the bias matrix, which is just a single row matrix. We use this comma
    notation, to tell numpy to perform broadcasting and I've covered
    broadcasting in my previous blogposts.
</p>
<Figure path="Forward Propagation 1.avif" caption="Forward Propagation Step" />
<p>
    The output of the convolutional layer is another set of matrices. While the
    input had <Katex math={"c_{in}"} /> as the number of channels, for our output,
    the number of channels is <Katex math={"c_{out}"} />. Also, each box here is
    a matrix, with the shape <Katex math={"p \\times q"} />, and these values
    depend on the width and height of our input matrices as well as our kernels.
    More specifically, what you see here is the resulting shape of a so-called
    valid cross-correlation between two matrices.
</p>
<p>
    For the actual forward propagation, we will take each of the input channel
    matrices and compute the cross-correlation between them and the matrices of
    the first kernel. Now you should see why each kernel must have exactly <Katex
        math={"c_{in}"}
    /> matrices, because each channel of the input needs to be cross-correlated with
    a kernel matrix. Then, we take the sum of each of these cross-correlations and
    add the bias on top. Remember, that the bias is a single number, but that's fine,
    since numpy will perform broadcasting for us and that single number will be added
    to every value of the matrix. The result of all of this is the first of the output
    matrices. We then repeat this process for every kernel.
</p>
<Figure path="fp1.avif" caption="Foward Propagation" />
