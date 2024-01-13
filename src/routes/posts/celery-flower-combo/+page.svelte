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
  </p>
</section>
<section>
  <h3>RabbitMQ</h3>
  <p>
    Now, we will setup RabbitMQ. We will use the official RabbitMQ docker image
    for this. Let's add this part to the <code>docker-compose.yml</code> file.
  </p>
  <CodeBox
    code={`
version: "3.8"
services:
  rabbitmq:
    image: rabbitmq:3-management
    hostname: rabbit_hostname
    container_name: rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "4369:4369"
      - "5671:5671"
      - "5672:5672"
      - "15671:15671"
      - "15691-15692:15691-15692"
      - "25672:25672"
      - "15672:15672"
  nginx:
    build:
      context: .
      dockerfile: Dockerfile.nginx
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - rabbitmq`}
    filename="docker-compose.yml"
    language="yaml"
  />
  <p>
    You will notice that I also forwarded some ports. I'm actually not sure if
    we even need all of them at this point. We will do an ablation study later.
    For now, I've just added all of them. We also want the Nginx container to
    start up last, so we add a <code>depends_on</code> field.
  </p>
  <p>
    Let's go ahead and add this to the <code>nginx.conf</code> file.
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
  upstream rabbitmq_management {
    server rabbitmq:15672;
  }

  server {
    listen 80;
    # server_name your_domain.com;
    
    # comment this if you want to use https
    location / {
      proxy_pass http://rabbitmq_management;
      proxy_set_header Host $http_host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Forwarded-Proto $scheme;
      proxy_read_timeout 300;
      proxy_connect_timeout 300;
      proxy_redirect off;
    }

    # uncomment this if you want to use https 
    # return 301 https://$host$request_uri;
  }
  
  # uncomment this if you want to use https
  # server {
  #   listen 443 ssl;
  #   server_name your_domain.com;
  #
  #   ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem; # Adjust to your certificate's path
  #   ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem; # Adjust to your key's path
  #   include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
  #
  #   location / {
  #     proxy_pass http://rabbitmq_management;
  #     proxy_set_header Host $http_host;
  #     proxy_set_header X-Real-IP $remote_addr;
  #     proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  #     proxy_set_header X-Forwarded-Proto $scheme;
  #     proxy_read_timeout 300;
  #     proxy_connect_timeout 300;
  #     proxy_redirect off;
  #   }
  # }
}
`}
    filename="nginx.conf"
    language="nginx"
  />
  <p>
    Running <code>docker compose up</code> should now start up both Nginx and
    RabbitMQ. If you navigate to
    <code>localhost:15672</code>, you should see the RabbitMQ management
    interface. You can login with the credentials we set in the
    <code>docker-compose.yml</code> file.
    <Figure path="rabbitmqsite.webp" caption="RabbitMQ Management" />
    That's all we need for now.
  </p>
</section>
<section>
  <h3>Postgres</h3>
  <p>
    Next up is Postgres. We will use the official Postgres docker image for
    this. Let's add this part to the <code>docker-compose.yml</code> file.
  </p>
  <CodeBox
    code={`
version: "3.8"
services:
  rabbitmq:
    image: rabbitmq:3-management
    hostname: rabbit_hostname
    container_name: rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "4369:4369"
      - "5671:5671"
      - "5672:5672"
      - "15671:15671"
      - "15691-15692:15691-15692"
      - "25672:25672"
      - "15672:15672"
  postgres:
    image: postgres:latest
    environment:
      POSTGRES_DB: main
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
  nginx:
    build:
      context: .
      dockerfile: Dockerfile.nginx
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - rabbitmq 
      - postgres 
`}
    filename="docker-compose.yml"
    language="yaml"
  />
</section>
<section>
  <h3>Flower</h3>
  <p>
    The next step is setting up Flower. We will use the official Flower docker
    image for that. Let's add that to our <code>docker-compose.yml</code> file.
  </p>
  <CodeBox
    code={`

version: "3.8"
services:
  rabbitmq:
    image: rabbitmq:3-management
    hostname: rabbit_hostname
    container_name: rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "4369:4369"
      - "5671:5671"
      - "5672:5672"
      - "15671:15671"
      - "15691-15692:15691-15692"
      - "25672:25672"
      - "15672:15672"
  postgres:
    image: postgres:latest
    environment:
      POSTGRES_DB: main
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
  flower:
    image: mher/flower
    environment:
      - FLOWER_PERSISTENT=True
      - FLOWER_DB=/etc/db/flower.db
    depends_on:
      - rabbitmq
    command: celery --broker=amqp://user:password@localhost:5672// flower --basic-auth=some_user:some_password
    ports:
      - "5555:5555"
    volumes:
      - ./flower/storage:/etc/db/
  nginx:
    build:
      context: .
      dockerfile: Dockerfile.nginx
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - rabbitmq 
`}
    filename="docker-compose.yml"
    language="yaml"
  />
  <p>
    We will also need to add some stuff to our <code>nginx.conf</code> file.
  </p>
  <CodeBox
    code={`
# ... the rest of the file
  upstream flower {
    server flower:5555;
  }

  server {
    listen 80;
    # comment this if you want to use https
    # server_name your_domain.com;
    
    # comment this if you want to use https
    location / {
      limit_except GET POST OPTIONS {
        deny all;
      }
      proxy_pass_header Server;
      proxy_set_header Host $http_host;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_pass http://flower;

      if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
        add_header 'Access-Control-Max-Age' 86400;
        add_header 'Content-Type' 'text/plain charset=UTF-8';
        add_header 'Content-Length' 0;
        return 204;
      }
      add_header 'Access-Control-Allow-Origin' '*';
      add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
      add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
    }
    # uncomment this if you want to use https
    # return 301 https://$host$request_uri;
  }
  # uncomment this if you want to use https
  # server {
  #   listen 443 ssl;
  #   server_name your_domain.com;
  #   client_max_body_size 40m;
  #   ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem;
  #   ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem;
  #   include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
  #
  #   location / {
  #     limit_except GET POST OPTIONS {
  #       deny all;
  #     }
  #     proxy_pass_header Server;
  #     proxy_set_header Host $http_host;
  #     proxy_http_version 1.1;
  #     proxy_set_header Upgrade $http_upgrade;
  #     proxy_set_header Connection "upgrade";
  #     proxy_pass http://flower;
  #
  #     if ($request_method = 'OPTIONS') {
  #       add_header 'Access-Control-Allow-Origin' '*';
  #       add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
  #       add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
  #       add_header 'Access-Control-Max-Age' 86400;
  #       add_header 'Content-Type' 'text/plain charset=UTF-8';
  #       add_header 'Content-Length' 0;
  #       return 204;
  #     }
  #     add_header 'Access-Control-Allow-Origin' '*';
  #     add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
  #     add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
  #   }
  # }

  # ... the rest of the file
`}
    filename="nginx.conf"
    language="nginx"
  />
  <p>
    If you navigate to <code>localhost:5555</code>, you should see the Flower
    dashboard. You will have to login with the credentials we set in the
    <code>docker-compose.yml</code>
    file. Furthermore, if you try to perform a GET request to, e.g.,
    <code>localhost:5555/api/workers</code>, you should get a 401 error. This is
    good for us, because we don't want unauthorized users to make requests to
    our Flower API (which is the gateway to our Celery workers). By adding to
    the GET request a authorization header with the credentials we set in the
    <code>docker-compose.yml</code> file, we can make the request!
  </p>
</section>
<section>
  <h3>Monitoring: Grafana and Prometheus</h3>
  <p>
    Next up we will setup the monitoring tools for our Celery workers, which are
    Grafana and Prometheus. Let's use the official Grafana and Prometheus docker
    images for this and add that to our <code>docker-compose.yml</code> file.
  </p>
  <CodeBox
    code={`

version: "3.8"
services:
  rabbitmq:
    image: rabbitmq:3-management
    hostname: rabbit_hostname
    container_name: rabbitmq
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    ports:
      - "4369:4369"
      - "5671:5671"
      - "5672:5672"
      - "15671:15671"
      - "15691-15692:15691-15692"
      - "25672:25672"
      - "15672:15672"
  postgres:
    image: postgres:latest
    environment:
      POSTGRES_DB: main
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
  flower:
    image: mher/flower
    container_name: flower
    environment:
      - FLOWER_PERSISTENT=True
      - FLOWER_DB=/etc/db/flower.db
    depends_on:
      - rabbitmq
    command: celery --broker=amqp://user:password@localhost:5672// flower --basic-auth=some_user:some_password
    ports:
      - "5555:5555"
    volumes:
      - ./flower/storage:/etc/db/
  prometheus:
        container_name: prometheus
        volumes:
            - ./prometheus.yml:/etc/prometheus/prometheus.yml
        ports:
            - 9090:9090
        image: prom/prometheus
  grafana:
      container_name: grafana
      volumes:
          - ./grafana_storage:/var/lib/grafana
      ports:
          - 3000:3000
      image: grafana/grafana
  nginx:
    build:
      context: .
      dockerfile: Dockerfile.nginx
    volumes:
      - /etc/letsencrypt:/etc/letsencrypt
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - rabbitmq
      - flower
      - prometheus 
      - grafana
`}
    filename="docker-compose.yml"
    language="yaml"
  />
  <p>
    We will also need to add some stuff to our <code>nginx.conf</code> file.
  </p>
  <CodeBox
    code={`
  # ... the rest of the file
  upstream prometheus {
    server prometheus:9090;
  }

  server {
    listen 80;
    # comment this if you want to use https
    location / {
      limit_except GET POST OPTIONS {
        deny all;
      }
      proxy_pass_header Server;
      proxy_set_header Host $http_host;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_pass http://prometheus;

      if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
        add_header 'Access-Control-Max-Age' 86400;
        add_header 'Content-Type' 'text/plain charset=UTF-8';
        add_header 'Content-Length' 0;
        return 204;
      }
      add_header 'Access-Control-Allow-Origin' '*';
      add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
      add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
    }
    # server_name your_domain.com;
    
    # uncomment this if you want to use https
    # return 301 https://$host$request_uri;
  }

  # server {
  #   listen 443 ssl;
  #   server_name your_domain.com;
  #   client_max_body_size 40m;
  #   ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem;
  #   ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem;
  #   include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
  #
  #   location / {
  #     limit_except GET POST OPTIONS {
  #       deny all;
  #     }
  #     proxy_pass_header Server;
  #     proxy_set_header Host $http_host;
  #     proxy_http_version 1.1;
  #     proxy_set_header Upgrade $http_upgrade;
  #     proxy_set_header Connection "upgrade";
  #     proxy_pass http://prometheus;
  #
  #     if ($request_method = 'OPTIONS') {
  #       add_header 'Access-Control-Allow-Origin' '*';
  #       add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
  #       add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
  #       add_header 'Access-Control-Max-Age' 86400;
  #       add_header 'Content-Type' 'text/plain charset=UTF-8';
  #       add_header 'Content-Length' 0;
  #       return 204;
  #     }
  #     add_header 'Access-Control-Allow-Origin' '*';
  #     add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
  #     add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
  #   }
  # }

  upstream grafana {
    server grafana:3000;
  }

  server {
    listen 80;
    
    # comment this if you want to use https
    location / {
      proxy_pass_header Server;
      proxy_set_header Host $http_host;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_pass http://grafana;

      if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
        add_header 'Access-Control-Max-Age' 86400;
        add_header 'Content-Type' 'text/plain charset=UTF-8';
        add_header 'Content-Length' 0;
        return 204;
      }
      add_header 'Access-Control-Allow-Origin' '*';
      add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
      add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
    }
    # uncomment this if you want to use https
    # server_name your_domain.com;
    # return 301 https://$host$request_uri;
  }

  # server {
  #   listen 443 ssl;
  #   server_name your_domain.com;
  #   client_max_body_size 40m;
  #   ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem;
  #   ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem;
  #   include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
  #
  #   location / {
  #     proxy_pass_header Server;
  #     proxy_set_header Host $http_host;
  #     proxy_http_version 1.1;
  #     proxy_set_header Upgrade $http_upgrade;
  #     proxy_set_header Connection "upgrade";
  #     proxy_pass http://grafana;
  #
  #     if ($request_method = 'OPTIONS') {
  #       add_header 'Access-Control-Allow-Origin' '*';
  #       add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
  #       add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
  #       add_header 'Access-Control-Max-Age' 86400;
  #       add_header 'Content-Type' 'text/plain charset=UTF-8';
  #       add_header 'Content-Length' 0;
  #       return 204;
  #     }
  #     add_header 'Access-Control-Allow-Origin' '*';
  #     add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
  #     add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization';
  #   }
  # }
  # ... the rest of the file
`}
    filename="nginx.conf"
    language="nginx"
  />
  <p>
    We will also need to add a <code>prometheus.yml</code> file, which will be the
    configuration for Prometheus.
  </p>
  <CodeBox
    code={`
global:
  scrape_interval:     15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: prometheus
    static_configs:
      - targets: ['prometheus:9090']
  - job_name: flower
    static_configs:
      - targets: ['flower:5555']
`}
    filename="prometheus.yml"
    language="yaml"
  />
  <p>
    One thing to note is that since both Grafana and Prometheus are running
    based on the same <code>docker-compose.yml</code> file, we can use the
    container names as the hostnames for the <code>prometheus.yml</code> file.
    This is why we have
    <code>prometheus:9090</code> and <code>flower:5555</code> as the targets. We
    also create a couple of folders for Grafana and Prometheus to store their data
    in.
  </p>
  <HintBox
    title="Hint"
    content="It would also make since to protect these monitoring sites using Nginx. That's left as an exercise for the reader."
  />
  <p>
    Once you run <code>docker compose up</code>, you should be able to navigate
    to
    <code>localhost:3000</code> and see the Grafana dashboard. I would then
    highly suggest you read
    <a
      href="https://flower.readthedocs.io/en/latest/prometheus-integration.html#"
      >the tutorial provided from Flower</a
    >
    to setup the Prometheus and Grafana dashboards. When you setup the monitoring,
    the connection URL in the Grafana dashboard should be
    <code>http://prometheus:9090</code>
    rather than <code>http://localhost:9090</code>, since we are running
    everything using the same network and the container name is
    <code>prometheus</code>. Finally, it should look like this:
  </p>
  <Figure path="grafana.webp" caption="Grafana Dashboard" />
</section>
<section>
  <h3>Backend API</h3>
  <p>
    We've setup everything we need for the Celery workers, so now we can setup
    the business logic API. This doesn't have to be in Python, so just for
    demonstration purposes, we will use a backend written in <b>OCaml</b>,
    because why not.
  </p>
</section>
