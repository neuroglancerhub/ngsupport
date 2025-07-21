import sys
import logging

from flask import Flask
from flask_cors import CORS
from flask_compress import Compress


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

# Enable gzip compression for responses when client supports it
Compress(app)

# TODO: Limit origin list here: CORS(app, origins=[...])
#CORS(app, origins=[r'.*\.janelia\.org', r'neuroglancer-demo\.appspot\.com'], supports_credentials=True)
CORS(app)

#
# FIXME:
#   I'm sure this is not the idiomatic way to implement a big flask app.
#   Rather than implementing all of our endpoints here and then calling out
#   to separate files, there must be a simple and idiomatic way to implement
#   endpoints in separate files.
#


@app.route('/small-mesh')
def _small_mesh():
    from ngsupport.small_mesh import generate_and_store_mesh
    return generate_and_store_mesh()


@app.route('/svmesh/<server>/<uuid>/<instance>/info')
@app.route('/svmesh/<server>/<uuid>/<instance>/<segment_type>/info')
def _svmesh_info(server, uuid, instance, segment_type):
    from ngsupport.svmesh import svmesh_info
    return svmesh_info()


@app.route('/svmesh/<server>/<uuid>/<instance>/<segment_type>/<segment_id>:0')
def _svmesh_manifest(server, uuid, instance, segment_type, segment_id):
    from ngsupport.svmesh import svmesh_manifest
    return svmesh_manifest(segment_id)


@app.route('/svmesh/<server>/<uuid>/<instance>/<segment_type>/meshes/<int:segment_id>.ngmesh')
def _svmesh_ngmesh(server, uuid, instance, segment_type, segment_id):
    from ngsupport.svmesh import svmesh_ngmesh
    return svmesh_ngmesh(server, uuid, instance, segment_type, segment_id)


@app.route('/block-mesh', methods=['POST'])
def _block_mesh():
    from ngsupport.small_mesh import gen_block_mesh
    return gen_block_mesh()


@app.route('/locate-body')
def _locate_body():
    from ngsupport.locate_body import locate_body
    return locate_body()


@app.route('/neuronjson_segment_properties/<server>/<uuid>/<instance>/<label>/info')
@app.route('/neuronjson_segment_properties/<server>/<uuid>/<instance>/<label>/<altlabel>/info')
def _neuronjson_segment_properties_info(server, uuid, instance, label, altlabel=None):
    from ngsupport.neuronjson_segment_properties import neuronjson_segment_properties_info
    return neuronjson_segment_properties_info(server, uuid, instance, label, altlabel)


@app.route('/neuronjson_segment_tags_properties/<server>/<uuid>/<instance>/<tags>/info')
def _neuronjson_segment_tags_properties_info(server, uuid, instance, tags):
    from ngsupport.neuronjson_segment_properties import neuronjson_segment_tags_properties_info
    return neuronjson_segment_tags_properties_info(server, uuid, instance, tags)


@app.route('/neuronjson_segment_synapse_properties/<server>/<uuid>/<instance>/<int:n>/info')
def _neuronjson_segment_synapse_properties_info(server, uuid, instance, n):
    from ngsupport.neuronjson_segment_properties import neuronjson_segment_synapse_properties_info
    return neuronjson_segment_synapse_properties_info(server, uuid, instance, n)


@app.route('/neuronjson_segment_note_properties/<server>/<uuid>/<instance>/<propname>/<int:n>/info')
def _neuronjson_segment_note_properties_info(server, uuid, instance, propname, n):
    from ngsupport.neuronjson_segment_properties import neuronjson_segment_note_properties_info
    return neuronjson_segment_note_properties_info(server, uuid, instance, propname, n)


@app.route('/synapse_annotations/<server>/<uuid>/<instance>/info')
def _synapse_annotations_info(server, uuid, instance):
    from ngsupport.synapse_annotations import synapse_annotations_info
    return synapse_annotations_info(server, uuid, instance)

@app.route('/synapse_annotations/<server>/<uuid>/<instance>/<index_key>/<item_key>')
def _synapse_annotations(server, uuid, instance, index_key, item_key):
    from ngsupport.synapse_annotations import synapse_annotations_by_id, synapse_annotations_by_related_id

    if index_key == 'by_id':
        syn_id = int(item_key)
        return synapse_annotations_by_id(server, uuid, instance, syn_id)
    elif index_key.startswith('by_rel'):
        relationship = index_key[len('by_rel_'):]
        segment_id = int(item_key)
        return synapse_annotations_by_related_id(server, uuid, instance, relationship, segment_id)
    else:
        raise ValueError(f"Invalid index key: {index_key}")



if __name__ == "__main__":
    print("Debug launch on http://0.0.0.0:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
