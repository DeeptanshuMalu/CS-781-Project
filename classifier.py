import numpy as np
from dotmap import DotMap

from verifai.client import Client

try:
    import tensorflow as tf
except ModuleNotFoundError:
    import sys

    sys.exit("This functionality requires tensorflow to be installed")

from renderer.kittiLib import getLib
from renderer.generator import genImage
from model.modelNN import Model
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iteration_num",
        type=int,
        help="Iteration number of re-training process",
        default=1,
    )
    return parser.parse_args()


class Classifier(Client):
    def __init__(self, classifier_data):
        port = classifier_data.port
        bufsize = classifier_data.bufsize
        super().__init__(port, bufsize)
        self.sess = tf.compat.v1.Session()
        self.nn = Model()
        self.nn.init(
            classifier_data.graph_path, classifier_data.checkpoint_path, self.sess
        )
        self.lib = getLib()

    def simulate(self, sample):
        img, _ = genImage(self.lib, sample)
        yTrue = len(sample.cars)
        yPred = np.argmax(self.nn.predict(np.array(img))[0]) + 1
        res = {}
        res["yTrue"] = yTrue
        res["yPred"] = yPred

        return res


PORT = 8888
BUFSIZE = 4096

args = parse_args()

classifier_data = DotMap()
classifier_data.port = PORT
classifier_data.bufsize = BUFSIZE

if args.iteration_num == 1:
    classifier_data.graph_path = (
        "./data/car_detector/checkpoint/car-detector-model.meta"
    )
    classifier_data.checkpoint_path = "./data/car_detector/checkpoint/"
else:
    classifier_data.graph_path = (
        f"./data/checkpoints/iteration_{args.iteration_num - 1}/car-detector-model.meta"
    )
    classifier_data.checkpoint_path = (
        f"./data/checkpoints/iteration_{args.iteration_num - 1}/"
    )

client_task = Classifier(classifier_data)
while True:
    if not client_task.run_client():
        print("End of all classifier calls")
        break
