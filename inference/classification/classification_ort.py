# for resnet, vgg, mobilenet, squeezenet

import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

model_path = '../models/classification/resnet/resnet50-v1-12.onnx'
# model = onnx.load(model_path)

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# onnxruntime.InferenceSession(path/to/model, providers=['CUDAExecutionProvider']).
so = ort.SessionOptions()
so.enable_profiling = True
so.log_severity_level = 0
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"], sess_options=so)
# session = ort.InferenceSession(model_path, providers=[
#     ("CUDAExecutionProvider", {
#         "cudnn_conv_algo_search": "DEFAULT"
#     })
# ], sess_options=so)
# session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"], sess_options=so)

def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img
def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    session.end_profiling()
    a = np.argsort(preds)[::-1]
    print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))

# Enter path to the inference image below
img_path = 'kitten.jpg'
predict(img_path)

