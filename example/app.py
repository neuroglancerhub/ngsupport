import os

from flask import Flask

app = Flask(__name__)

@app.route('/')
@app.route('/<string:endpoint>')
def hello_world(endpoint=None):
    target = os.environ.get('TARGET', 'World')
    return f'Hello {target} from ({endpoint})!\n'

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
