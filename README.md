ngsupport
=========

Serverless utilties to support neuroglancer-based proofreading,
based on Google's "Cloud Run" service.

This directory contains the setup for a container that can be used with
Google's "Cloud Run" service to run various serverless cloud functions
that support our neuroglancer-based proofreading protocols, such as
generating meshes on-the-fly (and storing them to DVID).

To build and upload container with Google Cloudbuild registry:

    gcloud builds submit --tag gcr.io/flyem-private/ngsupport

To build FASTER using the most recent container as the cache:

    gcloud builds submit --config cloudbuild.yaml


Alteratively, just use docker to build locally:

    docker build --platform linux/amd64 . -t gcr.io/flyem-private/ngsupport

To push to the Google Artifact Registry, you need to authenticate with Google cloud and configure Docker use those credentials:

    gcloud auth login
    gcloud auth configure-docker     # first time only

Then:

    docker push gcr.io/flyem-private/ngsupport


NOTE: None of the above commands will actually DEPLOY the container.
      The easiest way to do that is via the google cloud console.

