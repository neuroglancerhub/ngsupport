import os
import logging
import datetime
import tempfile
import json
import urllib
from textwrap import dedent

from google.cloud import storage
from flask import Response, request, url_for, current_app

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
    blob.cache_control = 'public, no-store'
    blob.upload_from_string(blob_contents, content_type='application/json')
    return blob.public_url


def shortng():
    try:
        return _shortng()
    except Exception as ex:
        logger.error(ex)
        raise


def _shortng():
    from_slack = ('Slackbot' in request.headers.get('User-Agent'))
    logger.info(f"from_slack: {from_slack}")

    from_web = (request.form.get('client') == 'web')
    logger.info(f"from_web: {from_web}")

    if 'text' in request.form:
        # https://api.slack.com/interactivity/slash-commands#app_command_handling
        data = request.form['text'].strip()
    else:
        # For simple testing.
        data = request.data.decode('utf-8').strip()

    data = data.replace('`', '')

    logger.info(data)
    name_and_link = data.split(' ')
    if len(name_and_link) == 0:
        msg = "Error: No link provided"
        logger.error(msg)
        if from_slack:
            return Response(msg, 200)
        else:
            return Response(msg, 400)

    if len(name_and_link) == 1 or name_and_link[0] == '{':
        filename = request.form.get('filename', None)
        filename = filename or datetime.datetime.now().strftime('%Y-%m-%d.%H%M%S')
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
        msg = "Error: Filename must not contain spaces, and links must start with http or https"
        logger.error(msg)
        if from_slack:
            return Response(msg, 200)
        else:
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

    url = f'{url_base}#!gs://{SHORTNG_BUCKET}/short/{filename}'
    logger.info(f"Completed {url}")

    if not from_web:
        return Response(url, 200)

    script = dedent("""
        function copy_to_clipboard(text) {
            try {
                navigator.clipboard.writeText(text);
            }
            catch (err) {
                console.error("Couldn't write to clipboard:", err)
            }
        }
        """)

    page = dedent(f"""\
        <html>
        <head>
        <title>Shortened link</title>
        <script type="text/javascript">
        {script}
        </script>
        </head>
        <body>
        <h2>
            <a href={url}>{url}</a>
            <a href="" onclick="copy_to_clipboard('{url}'); return false;">
                <img src=static/copy.jpg width=30 height=30>
            </a>
        </h2>
        <h3><a href=shortener.html>[Start Over]</a></h3>
        </body>
        </html>
        """)
    return Response(page, 200)


def shortener():
    return current_app.send_static_file('shortener.html')
