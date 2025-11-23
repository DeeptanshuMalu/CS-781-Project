"""PyTorch Model trainer"""

from model import dataset, utils
from model.pytorch_model import CarDetectorModel
from model.modelNN import Model

import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import os
import numpy as np
from tqdm import tqdm
import random
import shutil

# Adding seed for consistent initialization
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation",
    )
    parser.add_argument(
        "--iteration_num",
        type=int,
        default=0,
        help="Iteration number of the re-training process",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=200,
        help="Number of training samples per class",
    )
    parser.add_argument(
        "--normal_training",
        action="store_true",
        help="Flag to indicate normal training without re-training iterations",
    )
    return parser.parse_args()


def convert_pytorch_to_tf(pytorch_model, tf_session, classes, imgSize):
    """Convert PyTorch model weights to TensorFlow v1 model"""
    print("Converting PyTorch model to TensorFlow v1...")
    
    # Get PyTorch model state dict
    pytorch_state = pytorch_model.state_dict()
    
    # Create TensorFlow model
    tf_model = Model()
    x, layerFc2, yTrue, yTrueCls, yPred, yPredCls = tf_model.getGraph(len(classes))
    
    # Initialize all TensorFlow variables first
    tf_session.run(tf.global_variables_initializer())
    
    # Get TensorFlow variables
    tf_vars = tf.global_variables()
    
    # Mapping between PyTorch and TensorFlow layers
    weight_mapping = {
        'conv1.weight': 'Variable:0',
        'conv1.bias': 'Variable_1:0',
        'conv2.weight': 'Variable_2:0',
        'conv2.bias': 'Variable_3:0',
        'conv3.weight': 'Variable_4:0',
        'conv3.bias': 'Variable_5:0',
        'fc1.weight': 'Variable_6:0',
        'fc1.bias': 'Variable_7:0',
        'fc2.weight': 'Variable_8:0',
        'fc2.bias': 'Variable_9:0',
    }
    
    # Convert and assign weights
    for pytorch_name, tf_name in weight_mapping.items():
        if pytorch_name in pytorch_state:
            pytorch_weight = pytorch_state[pytorch_name].cpu().numpy()
            
            # Find corresponding TensorFlow variable
            tf_var = None
            for var in tf_vars:
                if tf_name in var.name:
                    tf_var = var
                    break
            
            if tf_var is not None:
                if 'weight' in pytorch_name:
                    if len(pytorch_weight.shape) == 4:  # Conv weights
                        # PyTorch: (out_channels, in_channels, height, width)
                        # TensorFlow: (height, width, in_channels, out_channels)
                        pytorch_weight = pytorch_weight.transpose(2, 3, 1, 0)
                    elif len(pytorch_weight.shape) == 2:  # FC weights
                        # PyTorch: (out_features, in_features)
                        # TensorFlow: (in_features, out_features)
                        pytorch_weight = pytorch_weight.transpose()
                else:
                    # Biases remain the same
                    pass
                
                # Assign weight to TensorFlow variable
                tf_session.run(tf_var.assign(pytorch_weight))
                print(f"Converted {pytorch_name} -> {tf_name}: {pytorch_weight.shape}")
    
    return x, layerFc2, yTrue, yTrueCls, yPred, yPredCls


def train_pytorch_model():
    args = parse_args()
    random.seed(args.iteration_num + 42)
    normal_str = "_normal" if args.normal_training else ""
    
    # Training paths and model parameters
    pytorch_checkPointName = (
        f"data{normal_str}/checkpoints/iteration_{args.iteration_num}/car-detector-pytorch-model.pth"
    )
    tf_checkPointName = f"data{normal_str}/checkpoints/iteration_{args.iteration_num}/car-detector-model"
    trainPath = f"data{normal_str}/train/iteration_{args.iteration_num}/"
    os.makedirs(f"data{normal_str}/checkpoints/iteration_{args.iteration_num}", exist_ok=True)
    os.makedirs(trainPath, exist_ok=True)
    
    classes = ["0", "1"]
    imgSize = 32
    numChannels = 3
    validationSize = args.validation_size

    for num_cars in ['0', '1']:
        os.makedirs(trainPath + f"{num_cars}/", exist_ok=True)
        # num_orig_imgs_other_class = len([i for i in os.listdir(trainPath + f"{1-int(num_cars)}/") if i.endswith((".png", ".jpg"))])
        dump_imgs = os.listdir(f"data{normal_str}/train_dump/{num_cars}/")
        random.shuffle(dump_imgs)
        orig_imgs = [i for i in os.listdir(trainPath + f"{num_cars}/") if i.endswith((".png", ".jpg"))]
        random.shuffle(orig_imgs)
        num_orig_imgs = len(orig_imgs)
        if args.normal_training:
            selected_orig_imgs = dump_imgs[: args.train_size - 0]
        else:
            # selected_orig_imgs = orig_imgs[: args.train_size-num_orig_imgs]
            if num_orig_imgs >= args.train_size // 2:
                for orig_train_img in orig_imgs[args.train_size // 2:]:
                    os.remove(trainPath + f"{num_cars}/" + orig_train_img)
                selected_orig_imgs = dump_imgs[: args.train_size // 2]
            else:
                selected_orig_imgs = dump_imgs[: args.train_size - num_orig_imgs]
            # selected_orig_imgs = dump_imgs[: max(num_orig_imgs, num_orig_imgs_other_class)-num_orig_imgs]
        for orig_train_img in selected_orig_imgs:
            shutil.copy(
                f"data{normal_str}/train_dump/{num_cars}/{orig_train_img}",
                trainPath + f"{num_cars}/orig_{orig_train_img}",
            )

    # Load training and validation data
    data = dataset.readTrainSets(trainPath, imgSize, classes, validationSize=validationSize)

    print("Complete reading input data. Will Now print a snippet of it")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = CarDetectorModel(n_classes=len(classes), img_size=imgSize).to(device)
    if args.iteration_num != 0:
        # Load previous iteration model weights
        previous_model_path = (
            f"./data{normal_str}/checkpoints/iteration_{args.iteration_num - 1}/car-detector-pytorch-model.pth"
        )
        model.load_state_dict(torch.load(previous_model_path, map_location=device))
        print(f"Loaded model weights from {previous_model_path}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training parameters
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training phase
        steps_per_epoch = max(1, data.train.num_examples // args.batch_size)
        for step in tqdm(range(steps_per_epoch)):
            xBatch, yTrueBatch, _, _ = data.train.nextBatch(args.batch_size)
            
            # Convert to PyTorch tensors and transpose from TF format to PyTorch format
            # TF format: [batch, height, width, channels]
            # PyTorch format: [batch, channels, height, width]
            inputs = torch.FloatTensor(xBatch).permute(0, 3, 1, 2).to(device)
            labels = torch.LongTensor(np.argmax(yTrueBatch, axis=1)).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_steps = max(1, data.valid.num_examples // args.batch_size)
            for step in range(val_steps):
                xValidBatch, yValidBatch, _, _ = data.valid.nextBatch(args.batch_size)
                
                # Convert to PyTorch tensors and transpose from TF format to PyTorch format
                inputs = torch.FloatTensor(xValidBatch).permute(0, 3, 1, 2).to(device)
                labels = torch.LongTensor(np.argmax(yValidBatch, axis=1)).to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # # Early stopping
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     patience_counter = 0
        #     # Save best model
        #     torch.save(model.state_dict(), pytorch_checkPointName)
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}. "
        #               f"Validation loss did not improve for {patience} consecutive epochs.")
        #         break
    
    # Load best model
    torch.save(model.state_dict(), pytorch_checkPointName)
    print(f"PyTorch model saved to {pytorch_checkPointName}")
    
    # Convert and save TensorFlow v1 model
    print("Converting PyTorch model to TensorFlow v1...")
    
    # Create TensorFlow session
    tf_session = tf.Session()
    
    # Convert PyTorch weights to TensorFlow (this now handles initialization internally)
    x, layerFc2, yTrue, yTrueCls, yPred, yPredCls = convert_pytorch_to_tf(
        model, tf_session, classes, imgSize
    )
    
    # Create saver after variables are initialized and weights are set
    tf_saver = tf.train.Saver()
    tf_saver.save(tf_session, tf_checkPointName)
    print(f"TensorFlow v1 model saved to {tf_checkPointName}")
    
    # Close TensorFlow session
    tf_session.close()


if __name__ == "__main__":
    train_pytorch_model()