# FROM condaforge/miniforge3:24.9.2-0
FROM condaforge/miniforge3

# this container was originally based on centos:8; the condaforge image
#   is Ubuntu; not sure if the encoding and timezone lines are needed,
#   but I've left them in

# Set an encoding to make things work smoothly.
ENV LANG=en_US.UTF-8

RUN apt update -y \
    && apt upgrade -y \
    && apt install -y tar bzip2 curl \
    && apt clean

# Set timezone to EST/EDT
RUN ln -s /usr/share/zoneinfo/EST5EDT /etc/localtime


COPY configure-conda.sh /opt/docker/bin/configure-conda.sh
RUN /opt/docker/bin/configure-conda.sh

# Install packages
# FIXME: Use environment.yml
RUN . /opt/conda/etc/profile.d/conda.sh \
 && conda create -n flyem python=3.12 flask flask-cors gunicorn google-cloud-storage neuclease 'vol2mesh>=0.1.post24'

ENV FLYEM_ENV=/opt/conda/envs/flyem

# Ensure that flyem/bin is on the PATH
# FIXME: I suppose the more proper thing to do would be
#        to call 'conda activate' in a custom ENTRYPOINT script.
ENV PATH=${FLYEM_ENV}/bin:${PATH}

# Copy local code to the container image.
ENV APP_HOME=/ngsupport-home
WORKDIR $APP_HOME
COPY ngsupport ${APP_HOME}/ngsupport

ENV PYTHONPATH=${APP_HOME}

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
#
#   Note
#   ----
#
#   For some reason we're seeing an error:
#
#       Invalid ENTRYPOINT. [name: "gcr.io/flyem-private/ngsupport@sha256:cd0581de54ec430af44f959716379527ec1b116aa93602c14bc09ed6372c31cd" error: "Invalid command \"/bin/sh\": file not found" ].
#
#   And one way to fix it might be to use the 'exec' form of the CMD directive:
#   https://stackoverflow.com/questions/62158782/invalid-command-bin-sh-file-not-found
#
#   Unfortunately, that means we can't use environment variables ($FLYEM_ENV, $PORT).
#   So I'm hard-coding them for now.
#
#CMD exec ${FLYEM_ENV}/bin/gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 2 ngsupport.app:app

CMD ["/opt/conda/envs/flyem/bin/gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "2", "ngsupport.app:app"]

