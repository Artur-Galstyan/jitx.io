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
  <p>
    We will create a <code>docker-compose.yml</code> file, where we will put everything
    in.
  </p>
</section>
<section>
  <h3>Nginx</h3>
  <p>
    The first thing we will setup is Nginx, our reverse proxy. There are 2
    things we need: a <code>Dockerfile.nginx</code> file, which contains the
    dockerised Nginx, as well as the <code>nginx.conf</code> file, which is our Nginx
    configuration. Let's start with the Dockerfile.
  </p>

  <CodeBox
    code={`
FROM nginx:stable-alpine

# Remove default nginx configuration
RUN rm /etc/nginx/conf.d/default.conf

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose ports
EXPOSE 80
EXPOSE 443

# Run Nginx
CMD ["nginx", "-g", "daemon off;"]
`}
    filename="Dockerfile.nginx"
    language="Dockerfile"
  />
  <p>
    Along with this Dockerfile, we also need the <code>nginx.conf</code> file, so
    let's set it up. We will gradually add more to this file as we go along.
  </p>
  <CodeBox
    code={`
user nginx;
worker_processes 2;

error_log /var/log/nginx/error.log;
pid /var/run/nginx.pid;

events {
  worker_connections 1024;
  use epoll;
}
http {
  # we will add the stuff in here later
}
`}
    filename="nginx.conf"
    language="nginx"
  />
  <p>
    Now that we have our Nginx configuration, we can add it to our
    <code>docker-compose.yml</code> file.
  </p>

  <CodeBox
    code={`
version: "3.8"
services:
  nginx:
    build:
      context: .
      dockerfile: Dockerfile.nginx
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt
    ports:
      - "80:80"
      - "443:443"
`}
    filename="docker-compose.yml"
    language="yaml"
  />
  <p>
    We can now run <code>docker compose up</code> and see if everything works.
    If you go to <code>localhost</code> in your browser, you should see the default
    Nginx page.
  </p>
</section>
