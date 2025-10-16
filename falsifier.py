from verifai.features.features import *
from verifai.samplers.feature_sampler import *
from verifai.falsifier import generic_falsifier
from verifai.monitor import specification_monitor
from dotmap import DotMap
from renderer.generator import genImage
from renderer.kittiLib import getLib
import pickle
import argparse
import os
import shutil
from tqdm import tqdm
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--xPos_l", type=float, help="Lower bound of xPos", default=0)
    parser.add_argument("--xPos_u", type=float, help="Upper bound of xPos", default=1)
    parser.add_argument("--yPos_l", type=float, help="Lower bound of yPos", default=0)
    parser.add_argument("--yPos_u", type=float, help="Upper bound of yPos", default=1)
    parser.add_argument(
        "--num_cars", type=int, help="Number of cars", default=1, choices=[1, 2]
    )
    parser.add_argument(
        "--brightness_l", type=float, help="Lower bound of brightness", default=0.5
    )
    parser.add_argument(
        "--brightness_u", type=float, help="Upper bound of brightness", default=1
    )
    parser.add_argument(
        "--sharpness_l", type=float, help="Lower bound of sharpness", default=0
    )
    parser.add_argument(
        "--sharpness_u", type=float, help="Upper bound of sharpness", default=1
    )
    parser.add_argument(
        "--contrast_l", type=float, help="Lower bound of contrast", default=0.5
    )
    parser.add_argument(
        "--contrast_u", type=float, help="Upper bound of contrast", default=1.5
    )
    parser.add_argument("--color_l", type=float, help="Lower bound of color", default=0)
    parser.add_argument("--color_u", type=float, help="Upper bound of color", default=1)
    parser.add_argument(
        "--iteration_num",
        type=int,
        help="Iteration number of re-training process",
        default=1,
    )
    parser.add_argument(
        "--num_images", type=int, help="Number of images to be generated", default=10
    )
    parser.add_argument(
        "--sample_types",
        type=str,
        help="Type of sampling: random or kclosest",
        default="random",
        choices=["random", "kclosest"],
    )
    parser.add_argument(
        "--num_best_features",
        type=int,
        help="Number of best features to be returned",
        default=2,
    )

    return parser.parse_args()


args = parse_args()

# Sampling domain

carDomain = Struct(
    {
        "xPos": Box([args.xPos_l, args.xPos_u]),
        "yPos": Box([args.yPos_l, args.yPos_u]),
        "carID": Categorical(*np.arange(0, 37)),
    }
)

space = FeatureSpace(
    {
        "backgroundID": Feature(Categorical(*np.arange(0, 35))),
        #'cars': Feature(carDomain, lengthDomain=DiscreteBox([1, 2])),
        "cars": Feature(Array(carDomain, (args.num_cars,))),
        "brightness": Feature(Box([args.brightness_l, args.brightness_u])),
        "sharpness": Feature(Box([args.sharpness_l, args.sharpness_u])),
        "contrast": Feature(Box([args.contrast_l, args.contrast_u])),
        "color": Feature(Box([args.color_l, args.color_u])),
    }
)
sampler = FeatureSampler.randomSamplerFor(space)


class confidence_spec(specification_monitor):
    def __init__(self):
        def specification(traj):
            return bool(traj["yTrue"] == traj["yPred"])

        super().__init__(specification)


MAX_ITERS = 3 * args.num_images
PORT = 8888
MAXREQS = 5
BUFSIZE = 4096

falsifier_params = DotMap(
    n_iters=MAX_ITERS, compute_error_table=True, fal_thres=0.5, verbosity=1
)

server_options = DotMap(port=PORT, bufsize=BUFSIZE, maxreqs=MAXREQS)

falsifier = generic_falsifier(
    sampler=sampler,
    server_options=server_options,
    monitor=confidence_spec(),
    falsifier_params=falsifier_params,
)
falsifier.run_falsifier()

analysis_params = DotMap()
analysis_params.random_params.count = args.num_images
analysis_params.k_closest_params.k = args.num_images
analysis_params.pca = True
analysis_params.k_clusters_params.k = 4
falsifier.analyze_error_table(analysis_params=analysis_params)
lib = getLib()

if args.iteration_num == 0:
    save_dir = f"data/test/{args.num_cars}"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
else:
    save_dir = f"data/train/iteration_{args.iteration_num}/{args.num_cars}"
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

print("Error table")
print(falsifier.error_table.table)
falsifier.error_table.table.to_csv(f"{save_dir}/error_table.csv")
print("Results of error table analysis")
if args.sample_types == "random":
    print("Random samples from error table")
    for i, sample in tqdm(
        enumerate(falsifier.error_analysis.random_samples),
        total=len(falsifier.error_analysis.random_samples),
    ):
        # print(sample)
        img, _ = genImage(lib, sample)
        img.save(f"{save_dir}/" + str(i) + ".png")
        # img.show()

elif args.sample_types == "kclosest":
    print("k closest samples from error table")
    for i, sample in tqdm(
        enumerate(falsifier.error_analysis.k_closest_samples),
        total=len(falsifier.error_analysis.k_closest_samples),
    ):
        # print(sample)
        img, _ = genImage(lib, sample)
        img.save(f"{save_dir}/" + str(i) + ".png")
        # img.show()

# print("k means clustering centroids from error table")
# print("Centroids for the categorical parts of the sample")
# print(falsifier.error_analysis.k_clusters.keys())

# print("Centroids for the numerical parts of the sample for each discrete cluster")
# for k in falsifier.error_analysis.k_clusters.keys():
#     print(falsifier.error_analysis.k_clusters[k])

print("PCA analysis")
print("PCA pivot: ", falsifier.error_analysis.pca["pivot"])
print("Directions: ", falsifier.error_analysis.pca["directions"])
print("Columns", falsifier.error_analysis.pca["columns"])

best_features_indexes = np.argsort(
    np.abs(falsifier.error_analysis.pca["directions"][0])
)[-args.num_best_features :]
best_columns = [
    falsifier.error_analysis.pca["columns"][i] for i in best_features_indexes
]
print(f"Best {args.num_best_features} features: ", best_columns, best_features_indexes)

with open(f"{save_dir}/best_features.json", "w") as f:
    json.dump(best_columns, f)

# To save all samples: uncomment this
# pickle.dump(falsifier.samples, open("generated_samples.pickle", "wb"))
# print(falsifier.samples)
# for i in falsifier.samples:
#     print(falsifier.samples[i])
#     img, _ = genImage(lib, falsifier.samples[i])
#     img.save(f"test.png")
#     # img.show()
#     break
