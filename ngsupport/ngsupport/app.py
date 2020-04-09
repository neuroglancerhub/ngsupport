import os
import logging

import numpy as np

from flask import Flask, Response, request, make_response
from flask_cors import CORS
from requests import RequestException

from neuclease import configure_default_logging
configure_default_logging()
logger = logging.getLogger(__name__)

from .meshgen import generate_and_store_mesh

app = Flask(__name__)

# TODO: Limit origin list here: CORS(app, origins=[...])
CORS(app)

@app.route('/smallmesh')
def smallmesh():
    return generate_and_store_mesh()

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
