import sys
import logging

from flask import Flask
from flask_cors import CORS


def configure_default_logging():
    """
    Simple logging configuration.
    Useful for interactive terminal sessions.
    """
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

    logging.captureWarnings(True)


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

@app.route('/shortener.html')
def _shortener():
    from ngsupport.shortng import shortener
    return shortener()

@app.route('/neuronjson_segment_properties/<server>/<uuid>/<instance>/<label>/<altlabel>/info')
def _neuronjson_segment_properties_info():
    from ngsupport.neuronjson_segment_properties import neuronjson_segment_properties_info
    return neuronjson_segment_properties_info()


if __name__ == "__main__":
    print("Debug launch on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
