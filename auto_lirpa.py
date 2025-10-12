from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration_num",
        type=int,
        help="Iteration number of re-training process",
        default=1,
    )
    return parser.parse_args()


# Parse arguments
args = parse_args()


# Determine device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Loading model on {device}")



# Load and preprocess test image
image_path = f"data/train/iteration_{args.iteration_num}/1/random_3.png"
# image_path = f"data/train/iteration_{args.iteration_num}/1/kclosest_0.png"
if not os.path.exists(image_path):
    # Create a dummy image for testing
    image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    print("Using dummy image for testing")
else:
    image = cv2.imread(image_path)
    print(f"Loaded image from {image_path}")

try:
    image = cv2.resize(image, (128, 128), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)

    # Convert from BGR to RGB and change from HWC to CHW format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))  # Change from HWC to CHW

except Exception as e:
    print(f"Issue with image preprocessing: {e}")
    exit(1)

# TENSORFLOW MODEL TESTING -----------------------------
import tensorflow as tf
from model.modelNN import Model

graph_path = (
    f"./data/checkpoints/iteration_{args.iteration_num}/car-detector-model.meta"
)
checkpoint_path = f"./data/checkpoints/iteration_{args.iteration_num}/"
sess = tf.compat.v1.Session()
nn = Model()
nn.init(graph_path, checkpoint_path, sess)
print("tf pred:", nn.predict(np.array(image)))
# TENSORFLOW MODEL TESTING ---------------------------------

# Create tensor with proper shape [batch, channels, height, width]
image_tensor = torch.tensor(image, device=device, dtype=torch.float32).unsqueeze(0)
print(f"Input image shape: {image_tensor.shape}")

# Test the model first
with torch.no_grad():
    test_output = model(image_tensor)
    # print(f"Test output shape: {test_output.shape}")
    print(f"Test softmax: {torch.softmax(test_output, dim=1)}")
    print(f"Test output: {test_output}")
    predicted_class = torch.argmax(test_output, dim=1)
    # print(f"Predicted class: {predicted_class.item()}")

# Create BoundedModule with the PyTorch model
bounded_model = BoundedModule(model, image_tensor)
bounded_model.eval()

# Set perturbation parameters
eps = 0.003
norm = np.inf
ptb = PerturbationLpNorm(norm=norm, eps=eps)

# Input tensor is wrapped in a BoundedTensor object
bounded_image = BoundedTensor(image_tensor, ptb)

# Get model prediction
prediction = bounded_model(bounded_image)
print("Model prediction:", prediction)

print("Computing bounds using CROWN method...")
with torch.no_grad():
    lb, ub = bounded_model.compute_bounds(x=(bounded_image,), method="CROWN")


# Print bounds
def print_bounds(lb, ub):
    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()
    for j in range(lb.shape[1]):
        print(
            "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f}".format(
                j=j, l=lb[0][j], u=ub[0][j]
            )
        )


print_bounds(lb, ub)
