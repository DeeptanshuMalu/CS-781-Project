# CS 781 Project

## Setup
Run the following to setup the project
```bash
pip install -r requirememnts.txt
pip install git+https://github.com/KaidiXu/auto_LiRPA.git
```

## Running instructions
We have created a complete pipeline of all the steps mentioned in the report. Run the following to start the re-training procedure
```bash
python main.py --total_iteration 10 --start_from_iteration 1
```
The `total_iteration` argument controls how many re-training iterations are run. The `start_from_iteration` argument controls from which re-training iteration to continue (if this is 0, procedure starts from pre-training).

## Generated Results
- In the `data/checkpoints` directory, you can find the models for all iterations.
- In `data/test/0` and `data/test/1`, you can find the test set images for each class.
- In `data/test/0/best_features` and `data/test/1/best_features`, you can find the values of the chosen high-level features on the the bottom 20% least robust images.
- In `data/test/0/robustness` and `data/test/1/robustness`, you can find the average robustness metric, average accuracy and individual robustness metirc values for the test images
- In `data/train`, you can find the training data for each iteration.

You can also run `plot_robustness.ipynb` to get the average robustness and average accuracy plots.