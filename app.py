from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

if __name__ == '__main__':
    from play import play_bp, frame, set_socket
    set_socket(socketio)

    app.register_blueprint(play_bp)
    @app.route('/')
    def hello_world():
        return 'Hello World'

    @socketio.on('frame')
    def on_frame(data):
        frame(data)

    # app.run(debug=True)
    # socketio.run(app, debug=False, ssl_context=('./server.crt', './server.key'))
    # app.run(host='0.0.0.0', port=8000, debug=False, ssl_context=('./server.crt', './server.key'))
    app.run(host='0.0.0.0', port=8000, debug=False)
