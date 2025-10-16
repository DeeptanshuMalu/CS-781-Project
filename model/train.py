"""Model trainer"""

from model import dataset, utils
from model.modelNN import Model
from model.pytorch_model import CarDetectorModel, convert_tf_to_pytorch

import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import argparse
import os

# Adding seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow.compat.v1 import set_random_seed

set_random_seed(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=500, help="Number of training iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer"
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
        default=1,
        help="Iteration number of the re-training process",
    )
    return parser.parse_args()


args = parse_args()

batchSize = args.batch_size

# Training paths and model parameters
checkPointName = f"data/checkpoints/iteration_{args.iteration_num}/car-detector-model"
pytorch_checkPointName = (
    f"data/checkpoints/iteration_{args.iteration_num}/car-detector-pytorch-model.pth"
)
trainPath = f"data/train/iteration_{args.iteration_num}/"
os.makedirs(f"data/checkpoints/iteration_{args.iteration_num}", exist_ok=True)
os.makedirs(trainPath, exist_ok=True)
classes = ["1", "2"]  # training folder names
imgSize = 128
numChannels = 3

# Validation %
validationSize = args.validation_size

# Load training and validation images and labels
data = dataset.readTrainSets(trainPath, imgSize, classes, validationSize=validationSize)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()

nn = Model()
x, layerFc2, yTrue, yTrueCls, yPred, yPredCls = nn.getGraph(len(classes))

session.run(tf.global_variables_initializer())

crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=layerFc2, labels=yTrue)
cost = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(cost)
correct_prediction = tf.equal(yPredCls, yTrueCls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

totalIterations = 0

saver = tf.train.Saver()

session.run(tf.global_variables_initializer())

if args.iteration_num == 1 or args.iteration_num == 0:
    nn.init(
        "data/car_detector/checkpoint/car-detector-model.meta",
        "data/car_detector/checkpoint",
        session,
    )
else:
    nn.init(
        f"data/checkpoints/iteration_{args.iteration_num - 1}/car-detector-model.meta",
        f"data/checkpoints/iteration_{args.iteration_num - 1}/",
        session,
    )


def showProgress(epoch, feedDictTrain, feedDictValidate, valLoss):
    acc = session.run(accuracy, feed_dict=feedDictTrain)
    val_acc = session.run(accuracy, feed_dict=feedDictValidate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, valLoss))


def train(numIteration):

    global totalIterations

    if args.iteration_num > 0:
        for i in range(totalIterations, totalIterations + numIteration):

            xBatch, yTrueBatch, _, _ = data.train.nextBatch(batchSize)
            xValidBatch, yValidBatch, _, _ = data.valid.nextBatch(batchSize)

            feedDictTr = {x: xBatch, yTrue: yTrueBatch}
            feedDictVal = {x: xValidBatch, yTrue: yValidBatch}

            session.run(optimizer, feed_dict=feedDictTr)

            if i % int(data.train.num_examples / batchSize) == 0:
                valLoss = session.run(cost, feed_dict=feedDictVal)
                epoch = int(i / int(data.train.num_examples / batchSize))

                showProgress(epoch, feedDictTr, feedDictVal, valLoss)
                saver.save(session, checkPointName)

        totalIterations += numIteration

    # Save TensorFlow model
    saver.save(session, checkPointName)
    print(f"TensorFlow model saved to {checkPointName}")

    # After TensorFlow training is complete, convert and save PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = CarDetectorModel(n_classes=len(classes)).to(device)

    # Convert TensorFlow weights to PyTorch
    pytorch_model = convert_tf_to_pytorch(
        session, tf.get_default_graph(), pytorch_model
    )

    # Save PyTorch model
    torch.save(pytorch_model.state_dict(), pytorch_checkPointName)
    print(f"PyTorch model saved to {pytorch_checkPointName}")


train(numIteration=args.num_iterations)
