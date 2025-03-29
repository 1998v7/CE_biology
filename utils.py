import time
import torch
import os
import json
import yaml
import argparse

class Timer:
    """
    A timer class to measure the time taken by a block of code.

    Usage:
    with Timer() as t:
        # code block
    print(f"Time taken: {t.elapsed_time} seconds")

    OR

    t = Timer()
    t.start()
    # code block
    t.end()
    print(f"Time taken: {t.elapsed_time} seconds")
    """

    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._elapsed_time = None

        # check if cuda is available
        self._on_gpu = torch.cuda.is_available()

    @property
    def elapsed_time(self):
        return self._elapsed_time
    
    def start(self):
        if self._on_gpu:
            torch.cuda.synchronize()
        self._start_time = time.time()

    def end(self):
        if self._on_gpu:
            torch.cuda.synchronize()
        self._end_time = time.time()
        self._elapsed_time = self._end_time - self._start_time
        return self._elapsed_time
    
    def reset(self):
        self._start_time = None
        self._end_time = None
        self._elapsed_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class MetricTracker:

    def __init__(self):
        self.metrics = dict(
            train_loss = [],
            train_acc = [],
            test_loss = [],
            test_acc = []
        )

    def update(self, name: str, value: float):
        self.metrics[name].append(value)

    def reset(self):
        for key in self.metrics:
            self.metrics[key] = []

    def to_json(self, path: str):
        # save to dict to json
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.metrics, f)

        
def read_config(path: str):
    # load the yaml file
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cath_topo.yaml")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    return parser.parse_args()