import copy

from flask import Response, request, make_response, jsonify
from requests import RequestException

from neuclease.dvid import default_dvid_session, find_master, generate_sample_coordinate


def locate_body():
    """
    Locate a body and return an arbitrary point that lies within it.

    Downloads the coarse sparsevol DVID, selects the "middle" (in the
    scan-order sense) block within the sparsevol representation,
    and downloads that block of segmentation at scale-0.
    Then selects the "middle" coordinate within the voxels of that
    block which match your body ID.

    Note: The returned point corresponds to the middle of the
    body's sparse representation, but is otherwise arbitrary.
    It is not the center of mass, nor is necessarily near the center
    of the body's bounding box.

    Pass DVID Authorization token via the 'Authorization' header,
    which will be forwarded to dvid requests.

    All other parameters should be passed via the query string.

    Parameters
    ----------
    dvid:
        dvid server. Required.

    uuid:
        Dvid uuid to read from and write to.

    segmentation:
        Name of the labelmap segmentation instance to read from.
        Default: 'segmentation'

    body:
        Body ID.

    user:
        The user name associated with this request.
        Will be forwarded to dvid requests.

    Returns:
        JSON [x,y,z]
    """
    try:
        dvid = request.args['dvid']
        body = request.args['body']
    except KeyError as ex:
        return Response(f"Missing required parameter: {ex.args[0]}", 400)

    body = int(body)

    uuid = request.args.get('uuid') or find_master(dvid)
    segmentation = request.args.get('segmentation', 'segmentation')
    supervoxels = request.args.get('supervoxels', 'false').lower()
    try:
        supervoxels = {'true': True, 'false': False}[supervoxels]
    except KeyError:
        return Response(f"Invalid argument for 'supervoxels' parameter: '{supervoxels}'")

    user = request.args.get('u') or request.args.get('user', "UNKNOWN")

    # TODO: The global cache of DVID sessions should store authentication info
    #       and use it as part of the key lookup, to avoid creating a new dvid
    #       session for every single cloud call!
    dvid_session = default_dvid_session('cloud-meshgen', user)
    auth = request.headers.get('Authorization')
    if auth:
        dvid_session = copy.deepcopy(dvid_session)
        dvid_session.headers['Authorization'] = auth

    coord_zyx = generate_sample_coordinate(dvid, uuid, segmentation, body, supervoxels, session=dvid_session)
    coord_xyz = coord_zyx.tolist()[::-1]
    return jsonify(coord_xyz)
