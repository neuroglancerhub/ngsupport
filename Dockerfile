# Note: This pixi version must be able to read our pixi.lock, whose format depends
# on the version of pixi that wrote it.  If you upgrade pixi itself, you may need
# to update this version, too, since newer versions of pixi may write lock files
# that older versions can't read.
FROM ghcr.io/prefix-dev/pixi:0.81.0 AS build

# copy source code, pixi.toml and pixi.lock to the container
COPY . /app
WORKDIR /app

RUN cat pixi.toml

# Install exactly the (linux-64) environment in pixi.lock.
# With --locked, the build fails if pixi.lock is out-of-date with pixi.toml, rather than
# re-solving the environment, so run 'pixi lock' (on any platform) after editing pixi.toml.
RUN pixi install --locked
RUN pixi list

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook --locked > /shell-hook.sh

# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /shell-hook.sh

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

# Use the PORT environment variable provided by Cloud Run, fallback to 8080 if not set.
# The number of worker processes and threads (per instance) can be configured via
# GUNICORN_WORKERS and GUNICORN_THREADS.  (Set the Cloud Run concurrency to their product.)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers ${GUNICORN_WORKERS:-4} --threads ${GUNICORN_THREADS:-2} ngsupport.app:app"]

