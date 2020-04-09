FROM centos:6

#MAINTAINER janelia-flyem <janelia-flyem@janelia.hhmi.org>

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8

# Resolves a nasty NOKEY warning that appears when using yum.
RUN rpm --import /etc/pki/rpm-gpg/RPM-GPG-KEY-CentOS-6

# Install basic requirements.
RUN yum update -y \
 && yum install -y \
            tar \
            bzip2 \
            curl \
 && yum clean all

COPY install-miniconda.sh /opt/docker/bin/install-miniconda.sh
RUN /opt/docker/bin/install-miniconda.sh

# Add a timestamp for the build. Also, bust the cache.
RUN date > container-build-date.txt

# Install packages
# FIXME: Use environment.yml
RUN source /opt/conda/etc/profile.d/conda.sh \
 && conda create -n flyem python=3.7 flask flask-cors gunicorn neuclease vol2mesh

ENV FLYEM_ENV /opt/conda/envs/flyem

# Ensure that flyem/bin is on the PATH
# FIXME: I suppose the more proper thing to do would be
#        to call 'conda activate' in a custom ENTRYPOINT script.
ENV PATH ${FLYEM_ENV}/bin:${PATH}

# Copy local code to the container image.
ENV APP_HOME /ngsupport-home
WORKDIR $APP_HOME
COPY ngsupport ${APP_HOME}

ENV PYTHONPATH=${APP_HOME}

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec ${FLYEM_ENV}/bin/gunicorn --bind :$PORT --workers 1 --threads 8 ngsupport:app
