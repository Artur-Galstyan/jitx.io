<script>
    import HintBox from "$lib/components/HintBox.svelte";
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
        those in a queue. It doesn't perform those tasks itself, but rather sends them to a celery worker using <i>messages</i>.
        When we refer to <i>Celery</i> we mean the "Celery App", which is running in tandem with our FastAPI app.
    </p>
    <p>
        RabbitMQ is a message broker. Whenever celery gets notified about a new task, it uses a message broker (such as
        RabbitMQ) which in turn sends the message it to a celery worker. The celery worker then performs the task and
        sends the result
        back to the message broker, which then sends it back to Celery.
    </p>
    <p>
        A celery worker is a process which performs whatever the task is. In our case, that's going to be a machine
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
</section>
