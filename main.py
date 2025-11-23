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
    parser.add_argument(
        "--start_from_iteration",
        type=int,
        help="Iteration number to start from",
        default=1,
    )
    parser.add_argument(
        "--normal_training",
        action="store_true",
        help="Flag to indicate normal training without re-training iterations",
    )
    return parser.parse_args()

args = parse_args()
normal_str = "_normal" if args.normal_training else ""

for train_path in os.listdir(f"data{normal_str}/train/"):
    if int(train_path.split("_")[-1]) < args.start_from_iteration:
        continue
    print("Clearing training data folder:", train_path)
    shutil.rmtree(os.path.join(f"data{normal_str}/train/", train_path))

for model_path in os.listdir(f"data{normal_str}/checkpoints/"):
    if int(model_path.split("_")[-1]) < args.start_from_iteration:
        continue
    print("Clearing checkpoints folder:", model_path)
    shutil.rmtree(os.path.join(f"data{normal_str}/checkpoints/", model_path))

for test_path in os.listdir(f"data{normal_str}/test/"):
    for iteration_path in os.listdir(os.path.join(f"data{normal_str}/test/", test_path)):
        if iteration_path == "best_features" or iteration_path == "robustness":
            for inner_path in os.listdir(os.path.join(f"data{normal_str}/test/", test_path, iteration_path)):
                if int(inner_path.split(".")[0].split("_")[-1]) < args.start_from_iteration:
                    continue
                print("Clearing test data folder:", inner_path)
                os.remove(os.path.join(f"data{normal_str}/test/", test_path, iteration_path, inner_path))

for iteration_num in range(args.start_from_iteration, args.total_iterations+1):
    #### RUN FALSIFIER AND CLASSIFIER ####
    if not args.normal_training:
        for num_cars in [0, 1]:
            if iteration_num not in [0, 1]:
                with open(f"data{normal_str}/test/{num_cars}/best_features/best_features_values_iteration_{iteration_num-1}.json", "r") as f:
                    best_features_values = json.load(f)

            port = get_free_port()
            print("Using port", port)

            # Run falsifier
            falsifier_args = ["python", "falsifier.py", "--iteration_num", str(iteration_num), "--port", str(port), "--num_cars", str(num_cars)]
            # if iteration_num != 0:
            #     for key in best_features_values:
            #         l, u = best_features_values[key]
            #         falsifier_args.extend([f"--{key}_l", str(l), f"--{key}_u", str(u)])
            falsifier = subprocess.Popen(falsifier_args)
            print("Falsifier started.")

            # Wait before starting classifier
            time.sleep(5)

            classifier_args = ["python", "classifier.py", "--iteration_num", str(iteration_num), "--port", str(port)]
            classifier = subprocess.Popen(classifier_args)
            print("Classifier started.")

            falsifier.wait()
            classifier.wait()

    #### RUN AUTO LIRPA ####
    if iteration_num != 0:
        for num_cars in [0, 1]:
            auto_lirpa_args = ["python", "auto_lirpa.py", "--iteration_num", str(iteration_num), "--num_cars", str(num_cars)]
            if args.normal_training:
                auto_lirpa_args.append("--normal_training")
            auto_lirpa_process = subprocess.Popen(auto_lirpa_args)
            auto_lirpa_process.wait()
        
    #### RUN TRAINING ####
    train_args = ["python", "-m", "model.train", "--iteration_num", str(iteration_num)]
    if iteration_num == 0:
        train_args.extend(["--num_epochs", "10"])
    if args.normal_training:
        train_args.append("--normal_training")
    train_process = subprocess.Popen(train_args)
    train_process.wait()
