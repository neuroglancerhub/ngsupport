import copy

from flask import Response, request, make_response, jsonify
from requests import RequestException

from neuclease.dvid import default_dvid_session, find_master, generate_sample_coordinate

def locate_body():
    try:
        dvid = request.args['dvid']
        body = request.args['body']
    except KeyError as ex:
        return Response(f"Missing required parameter: {ex.args[0]}", 400)

    body = int(body)

    uuid = request.args.get('uuid') or find_master(dvid)
    segmentation = request.args.get('segmentation', 'segmentation')
    try:
        supervoxels = request.args.get('segmentation', 'false').lower()
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
