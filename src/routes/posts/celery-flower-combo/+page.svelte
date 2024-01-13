<script lang="ts">
  import HintBox from "$lib/components/HintBox.svelte";
  import Figure from "$lib/components/Figure.svelte";
  import CodeBox from "$lib/components/CodeBox.svelte";
</script>

<section>
  <h3>Introduction</h3>
  <p>
    I'm going to assume you already read my previous <a
      href="https://www.jitx.io/posts/ml-deployment">blogpost</a
    > but now wish to go one step further and also add monitoring and a working API.
    The following figure shows an overview of the setup we will try to build.
  </p>
  <Figure
    path="overview.drawio.svg"
    caption="Overview of the non-worker part"
  />
  <p>
    There's quite a lot we need to setup in our <code>docker-compose.yml</code> file,
    namely 7 services in total.
  </p>
  <ul>
    <li>
      <b>Grafana and Prometheus:</b> those will be our monitoring tools. Grafana
      will be what you will actually look and and Prometheus will provide the data
      for Grafana.
    </li>
    <li>
      <b>Flower</b> will talk with the Celery workers and provide us with some very
      useful API features, such as being able to send tasks to the workers as well
      as check the tasks' status.
    </li>
    <li>
      <b>Postgres</b> will be the result backend for the Celery workers.
    </li>
    <li>
      <b>RabbitMQ</b> will be the message broker for the Celery workers.
    </li>
    <li>
      <b>Nginx</b> is the reverse proxy behind which everything will run. We will
      use it to protect our API from unauthorized users.
    </li>
    <li>
      <b>Backend API:</b> this will be any backend which will hold our business logic.
    </li>
  </ul>
  <p>
    The important part is that the <b>backend API</b> will send tasks to the
    workers via <b>Flower</b>, which provides a RESTful API. This means our
    backend API doesn't have to be written in Python! If Flower didn't exist, we
    would have to import the Celery app and use the `send_task` function of the
    Celery app object, restricting us to use Python as our backend. With Flower,
    we can just send a post request to the Flower API, which will forward our
    request to the Celery workers.
  </p>
  <Figure path="task_flow.drawio.svg" caption="Task Flow" />
  <p>
    Flower, under the hood, is just a <a
      href="https://github.com/tornadoweb/tornado">Python Tornado</a
    > application, which instantiates the Celery app object and then simply calls
    the `send_task` function to create a task for the workers. Once the workers are
    done, the result is written in the result backend.
  </p>
  <HintBox
    title="Hint"
    content="You could also use tools like Supabase and leverage the realtime capabilities to provide your users the task result as soon as it finishes!"
  />
  <p>Let's start to set everything up!</p>
</section>
