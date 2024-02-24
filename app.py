from flask import Flask, render_template
from flask_socketio import SocketIO
import torch

app = Flask(__name__)
socketio = SocketIO(app)

model = torch.jit.load('musclenet2.pt')
model.eval()

if __name__ == '__main__':
    from play import play_bp, frame, set_socket, set_model
    set_socket(socketio)
    set_model(model)
    
    app.register_blueprint(play_bp)

    @app.route('/')
    def hello_world():
        return render_template('splash.html')

    @socketio.on('frame')
    def on_frame(data):
        frame(data)

    # app.run(debug=True)
    # socketio.run(app, debug=False, ssl_context=('./server.crt', './server.key'))
    # app.run(host='0.0.0.0', port=8000, debug=False, ssl_context=('./server.crt', './server.key'))
    app.run(port=9000, debug=True)
