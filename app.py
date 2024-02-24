from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
from dataclasses import dataclass

app = Flask(__name__)
socketio = SocketIO(app)


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


@dataclass
class Config:
    task_name: str = 'multi_labeling'
    d_model: int = 96
    e_layers: int = 2
    dropout: float = 0.1
    enc_in: int = 34
    seq_len: int = 133
    num_class: int = 4
    embed: str = 'timeF'
    pred_len: int = 0
    label_len: int = 34
    top_k: int = 3
    d_ff: int = 32
    num_kernels: int = 6
    freq: str = 'h'
    n_heads: int = 8


# Load the trained model
config = Config()
model = torch.jit.load('musclenet.pt')
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
