from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import os
# import tensorflow as tf
from model.pytorch_model import CarDetectorModel
from model.modelNN import Model

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration_num",
        type=int,
        help="Iteration number of re-training process",
        default=1,
    )
    parser.add_argument(
        "--num_cars",
        type=int,
        default=1,
        help="Number of cars in the image",
        choices=[1, 2],
    )
    return parser.parse_args()


# Parse arguments
args = parse_args()

# Load the PyTorch model
model = CarDetectorModel(n_classes=2)
model_path = (
    f"./data/checkpoints/iteration_{args.iteration_num}/car-detector-pytorch-model.pth"
)
if not os.path.exists(model_path):
    print(f"Model path {model_path} does not exist. Please run train.py first.")
    exit(1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


print(f"Loading model on {device}")

base_path = f"data/test/{args.num_cars}/"
for image_path in os.listdir(base_path):
    # Load and preprocess test image
    if not image_path.endswith((".jpg", ".png")):
        continue  # Skip non-image files
    print(f"Processing image: {os.path.join(base_path, image_path)}")
    robustness_values = []
    image = cv2.imread(os.path.join(base_path, image_path))

    try:
        image = cv2.resize(image, (128, 128), 0, 0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        print(f"Original image shape: {image.shape}")

        # Change from HWC to CHW format
        image = np.transpose(image, (2, 0, 1)).copy()  # Change from HWC to CHW

    except Exception as e:
        print(f"Issue with image preprocessing: {e}")
        exit(1)

    # TENSORFLOW MODEL TESTING -----------------------------
    # graph_path = (
    #     f"./data/checkpoints/iteration_{args.iteration_num}/car-detector-model.meta"
    # )
    # checkpoint_path = f"./data/checkpoints/iteration_{args.iteration_num}/"
    # sess = tf.compat.v1.Session()
    # tf_model = Model()
    # tf_model.init(graph_path, checkpoint_path, sess)
    # print("TF output:", tf_model.predict(np.array(image)))
    # TENSORFLOW MODEL TESTING ---------------------------------

    # Create tensor with proper shape [batch, channels, height, width]
    image_tensor = torch.tensor(image, device=device, dtype=torch.float32).unsqueeze(0)
    print(f"Input image shape: {image_tensor.shape}")

    # Test the model first
    with torch.no_grad():
        test_output = model(image_tensor)
        # print(f"Test output shape: {test_output.shape}")
        print(f"Torch output: {torch.softmax(test_output, dim=1)}")
        predicted_class = torch.argmax(test_output, dim=1)
        is_correct = (predicted_class.item() + 1) == args.num_cars
        print(f"Predicted: {predicted_class.item() + 1}, True: {args.num_cars}, Correct: {is_correct}")

    # Create BoundedModule with the PyTorch model
    bounded_model = BoundedModule(model, image_tensor)
    bounded_model.eval()

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

    def find_max_robust_eps(
        bounded_model, image_tensor, pred_num_cars, max_eps=0.005, tolerance=1e-4
    ):
        """
        Binary search to find the largest epsilon that maintains robustness.

        Args:
            bounded_model: The bounded neural network model
            image_tensor: Input image tensor
            num_cars: Number of cars (1 or 2)
            max_eps: Maximum epsilon to search up to
            tolerance: Search tolerance

        Returns:
            float: Largest robust epsilon
        """

        def check_robustness(eps_val):
            norm = np.inf
            ptb = PerturbationLpNorm(norm=norm, eps=eps_val)
            bounded_image = BoundedTensor(image_tensor, ptb)

            with torch.no_grad():
                lb, ub = bounded_model.compute_bounds(
                    x=(bounded_image,), method="CROWN"
                )

            lb_np = lb.detach().cpu().numpy()[0]  # Remove batch dimension
            ub_np = ub.detach().cpu().numpy()[0]  # Remove batch dimension

            print_bounds(lb, ub)

            if pred_num_cars == 1:
                # For 1 car: lb[0] >= ub[1] (first neuron lower bound >= second neuron upper bound)
                return lb_np[0] >= ub_np[1]
            else:  # num_cars == 2
                # For 2 cars: lb[1] >= ub[0] (second neuron lower bound >= first neuron upper bound)
                return lb_np[1] >= ub_np[0]

        # Binary search
        low, high = 0.0, max_eps
        robust_eps = 0.0

        print(f"Starting binary search for robust epsilon (num_cars={pred_num_cars})...")

        while high - low > tolerance:
            mid = (low + high) / 2.0
            print(
                f"  Current bounds: low={low:.6f}, high={high:.6f} and trying mid={mid:.6f}"
            )

            if check_robustness(mid):
                robust_eps = mid
                low = mid
                print(f"  eps={mid:.6f}: ROBUST")
            else:
                high = mid
                print(f"  eps={mid:.6f}: NOT ROBUST")
            print("-" * 50)

        return robust_eps

    # Replace the existing bounds computation with the search
    print("Finding maximum robust epsilon...")
    max_robust_eps = find_max_robust_eps(bounded_model, image_tensor, predicted_class.item() + 1)
    print(f"Maximum robust epsilon: {max_robust_eps:.6f}")

    robustness_values.append(max_robust_eps if is_correct else -1 * max_robust_eps)

print("Robustness values for all images:")
print(robustness_values)
