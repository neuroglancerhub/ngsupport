FROM ghcr.io/prefix-dev/pixi:0.40.0 AS build

# copy source code, pixi.toml and pixi.lock to the container
COPY . /app
WORKDIR /app

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook > /shell-hook.sh

# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /shell-hook.sh

RUN cat pixi.toml
RUN pixi install
RUN pixi list

FROM ubuntu:24.04 AS production

# only copy the production environment into prod container
# please note that the "prefix" (path) needs to stay the same as in the build container
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /shell-hook.sh /shell-hook.sh

WORKDIR /app
EXPOSE 8080

# Set an encoding to make things work smoothly.
ENV LANG=en_US.UTF-8

# Set timezone to EST/EDT
RUN ln -s /usr/share/zoneinfo/EST5EDT /etc/localtime

ENV FLYEM_ENV=/app/.pixi/envs/default

ENV PATH=${FLYEM_ENV}/bin:${PATH}

# Copy local code to the container image.
ENV APP_HOME=/ngsupport-home
WORKDIR $APP_HOME
COPY ngsupport ${APP_HOME}/ngsupport

ENV PYTHONPATH=${APP_HOME}

# set the entrypoint to the shell-hook script (activate the environment and run the command)
# no more pixi needed in the prod container
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]

# Use the PORT environment variable provided by Cloud Run, fallback to 8080 if not set
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 4 --threads 2 ngsupport.app:app"]

