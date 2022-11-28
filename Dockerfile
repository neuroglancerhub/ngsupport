FROM centos:8

#MAINTAINER janelia-flyem <janelia-flyem@janelia.hhmi.org>

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8

# Resolves a nasty NOKEY warning that appears when using yum.
#RUN rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-8

# Install basic requirements.
RUN yum update -y \
 && yum install -y \
            tar \
            bzip2 \
            curl \
 && yum clean all

# Set timezone to EST/EDT
RUN rm /etc/localtime \
 && ln -s /usr/share/zoneinfo/EST5EDT /etc/localtime

COPY install-miniconda.sh /opt/docker/bin/install-miniconda.sh
RUN /opt/docker/bin/install-miniconda.sh

# Install packages
# FIXME: Use environment.yml
RUN source /opt/conda/etc/profile.d/conda.sh \
 && conda create -n flyem python=3.10 flask flask-cors gunicorn neuclease 'vol2mesh>=0.1.post20'

ENV FLYEM_ENV /opt/conda/envs/flyem

# Ensure that flyem/bin is on the PATH
# FIXME: I suppose the more proper thing to do would be
#        to call 'conda activate' in a custom ENTRYPOINT script.
ENV PATH ${FLYEM_ENV}/bin:${PATH}

# Copy local code to the container image.
ENV APP_HOME /ngsupport-home
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

