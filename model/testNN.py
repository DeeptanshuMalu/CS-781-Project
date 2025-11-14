import sys, getopt
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import torch

from model.modelNN import Model
import matplotlib.image as mpimg
from model.pytorch_model import CarDetectorModel
from PIL import Image
import numpy as np

iter_num = 0

GRAPH_PATH = f"./data/checkpoints/iteration_{iter_num}/car-detector-model.meta"
CHECKPOINT_PATH = f"./data/checkpoints/iteration_{iter_num}/"
IMAGE_PATH = "./data/test/0/19.png"

PYTORCH_MODEL_PATH = f"./data/checkpoints/iteration_{iter_num}/car-detector-pytorch-model.pth"

image = cv2.imread(IMAGE_PATH)
tf_session = tf.Session()

nn = Model()
nn.init(GRAPH_PATH, CHECKPOINT_PATH, tf_session)
print("TF:", nn.predict(image)[0])

pytorch_model = CarDetectorModel(n_classes=2, img_size=32)
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
pytorch_model.eval()

imageSize = 32
numChannels = 3
images = []
# image = cv2.resize(image, (imageSize, imageSize), cv2.INTER_LINEAR)
image = cv2.imread(IMAGE_PATH)
pix = Image.fromarray(image, "RGB")
pix = pix.resize((imageSize, imageSize), Image.Resampling.LANCZOS)
image = np.array(pix)

images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype("float32")
images = np.multiply(images, 1.0 / 255.0)

# Reshape for network input [None imageSize imageSize numChannels]
xBatch = images.reshape(1, imageSize, imageSize, numChannels)

# Convert to PyTorch tensor and change to NCHW format
xBatch = torch.from_numpy(xBatch).permute(0, 3, 1, 2)
with torch.no_grad():
    output = pytorch_model(xBatch)
    output = torch.softmax(output, dim=1)
    print("PyTorch:", output.numpy()[0])