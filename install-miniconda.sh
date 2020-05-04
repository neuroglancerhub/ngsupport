#!/bin/bash -il

set -exo pipefail

export condapkg="Miniconda3-py37_4.8.2-Linux-x86_64.sh"
export conda_chksum="87e77f097f6ebb5127c77662dfc3165e"

# Install the latest Miniconda with Python 3 and update everything.
curl -s -L https://repo.continuum.io/miniconda/$condapkg > miniconda.sh
openssl md5 miniconda.sh | grep $conda_chksum
bash miniconda.sh -b -p /opt/conda
rm -f miniconda.sh
touch /opt/conda/conda-meta/pinned
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate
conda config --set show_channel_urls True
conda config --add channels conda-forge
conda config --add channels flyem-forge
conda update --all --yes
conda clean -tipy
