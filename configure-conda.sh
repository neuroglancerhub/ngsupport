#!/bin/bash -il

# this script used to also install miniconda, but we're using an image
#   that has it preinstalled; I've left only the configuration lines here

set -exo pipefail

touch /opt/conda/conda-meta/pinned
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate
conda config --set show_channel_urls True
conda config --add channels nodefaults
conda config --add channels conda-forge
conda config --add channels flyem-forge
conda update --all --yes
conda clean -tipy
