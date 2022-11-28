import os
import logging
import datetime
import tempfile
import json
import urllib

from google.cloud import storage
from flask import Response, request

logger = logging.getLogger(__name__)


SHORTNG_BUCKET = 'flyem-user-links'  # Owned by FlyEM-Private


def parse_nglink(link):
    url_base, pseudo_json = link.split('#!')
    pseudo_json = urllib.parse.unquote(pseudo_json)
    data = json.loads(pseudo_json)
    return url_base, data


def upload_to_bucket(blob_name, blob_contents, bucket_name):
    storage_client = storage.Client.from_service_account_json(
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(blob_contents)
    return blob.public_url


def shortng():
    logger.info(request.data.decode('utf-8'))
    parsed = urllib.parse.parse_qs(request.data.decode('utf-8'))
    if parsed:
        # https://api.slack.com/interactivity/slash-commands#app_command_handling
        data = parsed['text'][0]
    else:
        # For simple testing.
        data = request.data.decode('utf-8').strip()

    name_and_link = data.split(' ')
    if len(name_and_link) == 0:
        msg = "No link provided"
        logger.error(msg)
        return Response(msg, 400)

    if len(name_and_link) == 1 or name_and_link[0] == '{':
        filename = datetime.datetime.now().strftime('%Y-%m-%d.%H%M%S')
        link = data
    else:
        filename = name_and_link[0]
        link = data[len(filename):].strip()

    if not filename.endswith('.json'):
        filename += '.json'

    try:
        state = json.loads(link)
        url_base = 'https://clio-ng.janelia.org'
    except ValueError:
        try:
            url_base, state = parse_nglink(link.strip())
        except ValueError:
            return Response(f"Could not parse link:\n\n{link}", 400)

    if not (url_base.startswith('http://') or url_base.startswith('https://')):
        msg = "Filename must not contain spaces, and links must start with http or https"
        logger.error(msg)
        return Response(msg, 400)

    # HACK:
    # I store the *contents* of the credentials in the environment
    # via the CloudRun settings, but the google API wants a filepath,
    # not a JSON string.
    # So I write the JSON text into a file before uploading.
    _fd, creds_path = tempfile.mkstemp('.json')
    with open(creds_path, 'w') as f:
        f.write(os.environ['GOOGLE_APPLICATION_CREDENTIALS_CONTENTS'])
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path

    state_string = json.dumps(state, indent=2)

    upload_to_bucket(f'short/{filename}', state_string, SHORTNG_BUCKET)

    return Response(f'{url_base}#!gs://{SHORTNG_BUCKET}/short/{filename}', 200)
