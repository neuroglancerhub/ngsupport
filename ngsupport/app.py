import os
import logging

import numpy as np

from flask import Flask, Response, request, make_response
from flask_cors import CORS
from requests import RequestException

from neuclease import configure_default_logging
configure_default_logging()
logger = logging.getLogger(__name__)

from .small_mesh import generate_and_store_mesh
from .locate_body import locate_body

app = Flask(__name__)

# TODO: Limit origin list here: CORS(app, origins=[...])
CORS(app, origins=r'.*\.janelia\.org', supports_credentials=True)
#CORS(app)

@app.route('/small-mesh')
def _small_mesh():
    return generate_and_store_mesh()

@app.route('/locate-body')
def _locate_body():
    return locate_body()


