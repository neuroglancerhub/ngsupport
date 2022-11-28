import logging

from flask import Flask
from flask_cors import CORS

from neuclease import configure_default_logging
configure_default_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

# TODO: Limit origin list here: CORS(app, origins=[...])
CORS(app, origins=r'.*\.janelia\.org', supports_credentials=True)
#CORS(app)


@app.route('/small-mesh')
def _small_mesh():
    from ngsupport.small_mesh import generate_and_store_mesh
    return generate_and_store_mesh()


@app.route('/block-mesh', methods=['POST'])
def _block_mesh():
    from ngsupport.small_mesh import gen_block_mesh
    return gen_block_mesh()


@app.route('/locate-body')
def _locate_body():
    from ngsupport.locate_body import locate_body
    return locate_body()


@app.route('/shortng', methods=['POST'])
def _shortng():
    from ngsupport.shortng import shortng
    return shortng()