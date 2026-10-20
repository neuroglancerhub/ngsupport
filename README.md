ngsupport
=========

Serverless utilties to support neuroglancer-based proofreading,
based on Google's "Cloud Run" service.

This directory contains the setup for a container that can be used with
Google's "Cloud Run" service to run various serverless cloud functions
that support our neuroglancer-based proofreading protocols, such as
generating meshes on-the-fly (and storing them to DVID).

Use docker to build locally:

    docker build --platform linux/amd64 . -t gcr.io/flyem-private/ngsupport

To push to the Google Artifact Registry, you need to authenticate with Google cloud and configure Docker use those credentials:

    gcloud auth login
    gcloud auth configure-docker     # first time only

Then:

    docker push gcr.io/flyem-private/ngsupport


NOTE: None of the above commands will actually DEPLOY the container.
      The easiest way to do that is via the google cloud console.


General hints: Colima setup
---------------------------

If you use [colima](https://github.com/abiosoft/colima) instead of Docker Desktop,
the `docker` commands here work the same way, with a few setup steps:

- Install colima, the `docker` command, and its `buildx` plugin (without which
  `docker build` falls back to the deprecated legacy builder):

      pixi global install colima docker-cli docker-buildx

  Make sure `~/.pixi/bin` comes before `/usr/local/bin` in your `PATH`, so you don't
  get a `docker` command left over from Docker Desktop.  (After installing, run `hash -r`
  or open a new shell, and check `type -a docker`.)

- Make the `buildx` plugin visible to docker, which looks for plugins in `~/.docker/cli-plugins`:

      mkdir -p ~/.docker/cli-plugins
      ln -sf ~/.pixi/envs/docker-buildx/bin/docker-buildx ~/.docker/cli-plugins/docker-buildx

  Check that it works with `docker buildx version`.  (The pixi build reports `v0.0.0+unknown`;
  a version ending in `-desktop` means docker is still using Docker Desktop's plugin.)

- On an Apple Silicon Mac, start the VM with Rosetta (to build the `linux/amd64` image
  without slow emulation) and enough memory for `pixi install` (the default 2 GB may not suffice):

      colima start --vm-type=vz --vz-rosetta --cpu 4 --memory 8

  (To change the settings of an existing VM, `colima delete` it and start it again.)

- If `~/.docker/config.json` still says `"credsStore": "desktop"` (left over from Docker Desktop),
  remove that line, or `docker push` will fail with "docker-credential-desktop not found".
  (Keep the `credHelpers` entries written by `gcloud auth configure-docker`.)

