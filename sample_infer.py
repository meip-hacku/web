import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession('./model/sample/sample.onnx')

input_data = np.random.rand(1, 10, 34).astype(np.float32)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

result = sess.run([output_name], {input_name: input_data})

print("result: ", result[0][0].tolist())