import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf


class CarDetectorModel(nn.Module):
    def __init__(self, n_classes=2):
        super(CarDetectorModel, self).__init__()

        # Network parameters (matching TF model)
        self.img_size = 128
        self.num_channels = 3
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Calculate the size after conv and pooling layers
        # Input: 128x128, after 3 max pools: 16x16, final feature maps: 64
        self.fc_input_size = 16 * 16 * 64

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

        # Initialize weights to match TF behavior
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights similar to TF truncated normal with stddev=0.05"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # TF uses truncated normal with stddev=0.05
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                # Clamp to simulate truncated normal (within 2 std devs)
                with torch.no_grad():
                    m.weight.clamp_(-0.1, 0.1)
                # TF uses constant 0.05 for biases
                nn.init.constant_(m.bias, 0.05)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                with torch.no_grad():
                    m.weight.clamp_(-0.1, 0.1)
                nn.init.constant_(m.bias, 0.05)

    def forward(self, x):
        # Conv layer 1 + MaxPool + ReLU (matching TF createConvolutionalLayer)
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        # Conv layer 2 + MaxPool + ReLU
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        # Conv layer 3 + MaxPool + ReLU
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)

        # Flatten layer (matching TF createFlattenLayer)
        # Use contiguous() before reshape to ensure memory layout compatibility
        x = x.contiguous().view(x.size(0), -1)

        # FC layer 1 + ReLU (matching TF createFcLayer with useRelu=True)
        x = self.fc1(x)
        x = F.relu(x)

        # FC layer 2 (matching TF createFcLayer with useRelu=False)
        x = self.fc2(x)

        return x

    def predict(self, image):
        """Predict single image (matching TF model's predict method)"""
        # Preprocessing to match TF model
        pix = Image.fromarray(image, "RGB")
        pix = pix.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        image = np.array(pix)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_tensor)
            # Apply softmax to get probabilities (matching TF model)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions.numpy()


def convert_tf_to_pytorch(tf_session, tf_graph, pytorch_model):
    """Convert TensorFlow weights to PyTorch model"""
    # Get TensorFlow variable values
    tf_vars = tf.global_variables()

    # Mapping from TF variable names to PyTorch parameters
    weight_mapping = {
        "Variable:0": "conv1.weight",  # Conv1 weights
        "Variable_1:0": "conv1.bias",  # Conv1 biases
        "Variable_2:0": "conv2.weight",  # Conv2 weights
        "Variable_3:0": "conv2.bias",  # Conv2 biases
        "Variable_4:0": "conv3.weight",  # Conv3 weights
        "Variable_5:0": "conv3.bias",  # Conv3 biases
        "Variable_6:0": "fc1.weight",  # FC1 weights
        "Variable_7:0": "fc1.bias",  # FC1 biases
        "Variable_8:0": "fc2.weight",  # FC2 weights
        "Variable_9:0": "fc2.bias",  # FC2 biases
    }

    pytorch_state_dict = pytorch_model.state_dict()

    for tf_var in tf_vars:
        tf_var_name = tf_var.name
        if tf_var_name in weight_mapping:
            pytorch_param_name = weight_mapping[tf_var_name]
            tf_value = tf_session.run(tf_var)

            # Convert TensorFlow weights to PyTorch format
            if "conv" in pytorch_param_name and "weight" in pytorch_param_name:
                # TF conv weights: [filter_height, filter_width, in_channels, out_channels]
                # PyTorch conv weights: [out_channels, in_channels, filter_height, filter_width]
                tf_value = np.transpose(tf_value, (3, 2, 0, 1))
            elif "fc" in pytorch_param_name and "weight" in pytorch_param_name:
                # TF fc weights: [in_features, out_features]
                # PyTorch fc weights: [out_features, in_features]
                tf_value = np.transpose(tf_value)

            # Update PyTorch parameter
            pytorch_state_dict[pytorch_param_name] = torch.from_numpy(tf_value)

    pytorch_model.load_state_dict(pytorch_state_dict)
    return pytorch_model
