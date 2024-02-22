import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tf2onnx

model = Sequential([
    LSTM(50, activation='relu', input_shape=(None, 34), name='lstm_layer'),
    Dense(4, name='dense_layer')
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.save('model/sample', save_format='tf')

spec = (tf.TensorSpec((None, None, 34), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, external_tensor_storage = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path, opset=13)