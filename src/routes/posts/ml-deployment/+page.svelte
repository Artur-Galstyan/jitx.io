<script>
    import HintBox from "$lib/components/HintBox.svelte";
    import Figure from "$lib/components/Figure.svelte";
</script>

<section>
    <h3>
        Introduction
    </h3>
    <p>
        Let's say you have trained a new model, which can totally change the world as we know it. All that's left is
        to deploy that model somewhere and let the world use it. But the model takes a while to perform inference, and
        if you deploy it on a single machine, the users' waiting time will continue to increase, which is obviously
        suboptimal.
    </p>
    <p>
        So, in this blog post, we will learn how to deploy your model using FastAPI, Celery and RabbitMQ. Let's get
        started!
    </p>
</section>
<section>
    <h3>
        What is FastAPI?
    </h3>
    <p>
        FastAPI is a modern, fast (high-performance), web framework for building APIs with Python. We can use it to
        accept requests from users and send back responses or - if we need real-time communication - use WebSockets.
    </p>
</section>
<section>
    <h3>What is Celery and RabbitMQ?</h3>
    <p>
        Celery is a task queue. In simple terms, it just receives notifications that a task has been requested and puts
        those in a queue. It doesn't perform those tasks itself, but rather sends them to a Celery worker using <i>messages</i>.
        When we refer to <i>Celery</i> we mean the "Celery App", which is running in tandem with our FastAPI app.
    </p>
    <p>
        RabbitMQ is a message broker. Whenever Celery gets notified about a new task, it uses a message broker (such as
        RabbitMQ) which in turn sends the message it to a Celery worker. The celery worker then performs the task and
        sends the result
        back to the message broker, which then sends it back to Celery.
    </p>
    <p>
        A Celery worker is a process which performs whatever the task is. In our case, that's going to be a machine
        learning
        model performing inference. Celery workers can be distributed across multiple machines, which means that we can
        scale our model to multiple machines and thus reduce the waiting time for users.
    </p>
    <p>
        To summarise, we have a FastAPI app, hosted on some machine to which we can send <kbd>POST</kbd> requests. The
        FastAPI app
        also contains our Celery app. When we send a <kbd>POST</kbd> request, the FastAPI app uses Celery (which in turn
        uses RabbitMQ) to send a message to a Celery worker. The Celery worker performs the task and stores the result
        in the
        "Result Store".
    </p>
    <HintBox title="Bonus Feature" content={`Many APIs these days use webhooks. When the user provides some endpoint
    and once our inference is done, we can send a <kbd>POST</kbd> request to that endpoint with the result. This notifies
    the user that the prediction is done.`}/>
</section>
<section>
    <h3>Creating the Model</h3>
    <p>
        Normally, I would have just inserted a placeholder task, like simply a delay of a couple of seconds to
        simulate a model performing inference. But I thought it would be more interesting to actually use a model. This
        part
        will be very brief, as it's not the main focus of this blog post.
    </p>
    <p>
        The model in question will be a CNN trained on the MNIST dataset. Our app will contain a drawing section, where
        the user can draw a number and our model will try to predict it.
    </p>
    <p>In the following collapsed section you can find the model and in the one after that the training code. It has 95%
        accuracy on MNIST. The model is saved in the <code>models</code> folder.
    </p>
    <div class="collapse border-accent border border-opacity-50">
        <input type="checkbox"/>
        <div class="collapse-title text-xl font-medium">
            The Model
        </div>
        <div class="collapse-content overflow-scroll">
<pre class="code">
    <code class="language-python">
        {`
import equinox as eqx
import jax.random
from jaxtyping import Array, PRNGKeyArray
from typing import Optional


class Model(eqx.Module):
    conv: eqx.nn.Conv2d
    mlp: eqx.nn.MLP
    max_pool: eqx.nn.MaxPool2d

    dropout: eqx.nn.Dropout

    def __init__(self):
        key, *subkeys = jax.random.split(jax.random.PRNGKey(33), 6)
        self.conv = eqx.nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=2, key=subkeys[1]
        )
        self.max_pool = eqx.nn.MaxPool2d(kernel_size=2)
        self.mlp = eqx.nn.MLP(
            in_size=2028, out_size=10, depth=3, width_size=128, key=subkeys[0]
        )
        self.dropout = eqx.nn.Dropout()

    def __call__(self, x: Array, key: Optional[PRNGKeyArray]):
        inference = True if key is None else False
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.dropout(x, key=key, inference=inference)
        x = x.ravel()
        x = self.mlp(x)
        x = jax.nn.softmax(x)
        return x

        `}
    </code>
</pre>
        </div>
    </div>
    <div class="collapse my-4 border-accent border border-opacity-50">
        <input type="checkbox"/>
        <div class="collapse-title text-xl font-medium ">
            The Training Code
        </div>
        <div class="collapse-content overflow-scroll">
<pre class="code">
    <code class="language-python">
        {`
import functools as ft
import time

import equinox as eqx
import jax.numpy as jnp
import jax.random
import optax
from icecream import ic
from jaxtyping import Array, PyTree, PRNGKeyArray, Float, Int
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model.model import Model


def main():
    # Transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.MNIST(
        "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
    )

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(trainset))
    validation_size = len(trainset) - train_size
    train_dataset, validation_dataset = random_split(
        trainset, [train_size, validation_size]
    )

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validationloader = DataLoader(validation_dataset, batch_size=128, shuffle=True)

    model = Model()
    optim = optax.adamw(learning_rate=0.001)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.key(45)
    start_time = time.time()
    for epoch in range(5):
        model = train(model, opt_state, optim, trainloader, key)
        eval_loss = evaluate(model, validationloader)
        ic(epoch, eval_loss)
    end_time = time.time()

    ic("training took " + str(end_time - start_time))
    eqx.tree_serialise_leaves("../models/model.eqx", model)


def loss_fn(
    model: PyTree,
    x: Float[Array, "batch 1 28 28"],
    y: Int[Array, " batch"],
    key: PRNGKeyArray,
) -> Float[Array, ""]:
    partial_model = ft.partial(model, key=key)
    pred_y = eqx.filter_vmap(partial_model)(x)

    loss = optax.softmax_cross_entropy(pred_y, y)

    return jnp.mean(loss)


@eqx.filter_jit
def step(
    model: PyTree,
    optim: optax.GradientTransformation,
    opt_state: PyTree,
    x: Array,
    y: Array,
    key: PRNGKeyArray,
):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


def one_hot_encode(labels, num_classes=10):
    return jnp.eye(num_classes)[labels]


def evaluate(model: PyTree, eval_dataloader: DataLoader):
    loss = 0
    accuracy = 0
    counter = 0
    acc_fn = lambda a, b: jnp.argmax(a) == b
    jitted_loss = eqx.filter_jit(loss_fn)
    for x, y in eval_dataloader:
        counter += len(x)
        x = x.numpy()
        target = y.numpy()
        y = one_hot_encode(y.numpy())
        loss += jitted_loss(model, x, y, key=None)

        pt_model = ft.partial(model, key=None)
        output = eqx.filter_vmap(pt_model)(x)
        accuracy += jnp.sum(jax.vmap(acc_fn)(output, target))

    return loss / counter, accuracy / counter


def train(
    model: PyTree,
    opt_state: PyTree,
    optim: optax.GradientTransformation,
    train_dataloader: DataLoader,
    key: PRNGKeyArray,
):
    for i, (x, y) in enumerate(train_dataloader):
        key, subkey = jax.random.split(key)
        x = x.numpy()
        y = one_hot_encode(y.numpy())
        model, opt_state, train_loss = step(model, optim, opt_state, x, y, subkey)
    return model


if __name__ == "__main__":
    main()

        `}
    </code>
</pre>
        </div>
    </div>
    <p>
        Alright, now we have a model trained and ready to go. What's next?
    </p>
</section>

<section>
    <h3>
        Deploying the Thing
    </h3>
    <p>
        Our architecture consists of multiple moving parts, which are illustrated in the following diagram:
    </p>
    <Figure path="Celery Workflow.drawio.svg" caption="The architecture of a Celery app" width={600}/>
    <p>
        Let's start with the FastAPI part, as that's pretty simple. Essentially, we need a FastAPI app with a
        <kbd>POST</kbd>
        endpoint which accepts an image and returns a task ID. The task ID is used to retrieve the status of the
        inference.
    </p>
    <HintBox content={`This is quite the overkill, because our inference only takes a new ms. But if you had a bigger
    model, such as a GAN or Stable Diffusion, then it makes sense to give the user access to the status of the
    inference.`}/>
</section>
