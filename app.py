from flask import Flask
from play import play_bp

app = Flask(__name__)
app.register_blueprint(play_bp)

@app.route('/')
def hello_world():
    return 'Hello World'

if __name__ == '__main__':
    app.run(debug=True)
