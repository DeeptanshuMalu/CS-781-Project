import subprocess
import time
import socket
import json
import argparse
import shutil
import os


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    addr, port = s.getsockname()
    s.close()
    return port

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total_iterations",
        type=int,
        help="Total number of re-training iterations",
        default=10,
    )
    return parser.parse_args()

args = parse_args()

for train_path in os.listdir("data/train/"):
    if train_path!="iteration_0":
        print("Clearing training data folder:", train_path)
        shutil.rmtree(os.path.join("data/train/", train_path))

for model_path in os.listdir("data/checkpoints/"):
    if model_path!="iteration_0":
        print("Clearing checkpoints folder:", model_path)
        shutil.rmtree(os.path.join("data/checkpoints/", model_path))

shutil.rmtree("data/test/0/best_features", ignore_errors=True)
shutil.rmtree("data/test/1/best_features", ignore_errors=True)
shutil.rmtree("data/test/0/robustness", ignore_errors=True)
shutil.rmtree("data/test/1/robustness", ignore_errors=True)

for iteration_num in range(1, args.total_iterations+1):
    for num_cars in [0, 1]:
        if iteration_num!=1:
            with open(f"data/test/{num_cars}/best_features/best_features_values_iteration_{iteration_num-1}.json", "r") as f:
                best_features_values = json.load(f)

        port = get_free_port()
        print("Using port", port)

        # Run falsifier
        falsifier_args = ["python", "falsifier.py", "--iteration_num", str(iteration_num), "--port", str(port), "--num_cars", str(num_cars)]
        if iteration_num != 1:
            for key in best_features_values:
                l, u = best_features_values[key]
                falsifier_args.extend([f"--{key}_l", str(l), f"--{key}_u", str(u)])
        falsifier = subprocess.Popen(falsifier_args)
        print("Falsifier started.")

        # Wait before starting classifier
        time.sleep(5)

        classifier_args = ["python", "classifier.py", "--iteration_num", str(iteration_num), "--port", str(port)]
        classifier = subprocess.Popen(classifier_args)
        print("Classifier started.")

        falsifier.wait()
        classifier.wait()

    train_args = ["python", "-m", "model.train", "--iteration_num", str(iteration_num)]
    train_process = subprocess.Popen(train_args)
    train_process.wait()

    for num_cars in [0, 1]:
        auto_lirpa_args = ["python", "auto_lirpa.py", "--iteration_num", str(iteration_num), "--num_cars", str(num_cars)]
        auto_lirpa_process = subprocess.Popen(auto_lirpa_args)
        auto_lirpa_process.wait()