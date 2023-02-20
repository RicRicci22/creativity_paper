import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class UAVCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images, questions = zip(*batch)
        images = torch.stack(images)
        # Might add encode batch
        list_tokenized_questions = [
            torch.tensor(
                self.tokenizer.encode(question, add_special_tokens=True).ids,
                dtype=torch.long,
            )
            for question in questions
        ]
        questions = pad_sequence(
            list_tokenized_questions, batch_first=True, padding_value=0
        )
        lengths = [sum(questions[i, :] != 0).item() for i in range(questions.shape[0])]
        index_sorted = sorted(
            range(len(lengths)), key=lambda k: lengths[k], reverse=True
        )
        lenghts = [lengths[i] for i in index_sorted]
        questions = questions[index_sorted]

        return images, questions, lenghts


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BeamSearchNode(object):
    def __init__(self, input, hiddenstate, previousNode, wordId, cumul_prob, length):
        """
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.input = input
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.cumul_prob = cumul_prob
        self.len = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return 1 - self.cumul_prob + alpha * reward


class custom_ln(torch.nn.Module):
    """
    Custom implementation of layer norm that takes into consideration the padding tokens, without using them to calculate the mean over the sequence.
    It is to be considered for NLP tasks, it normalizes along the embedding dimension for batches of sequences.
    Inputs:
    d_model: dimensionality of the input embeddings (int)
    """

    def __init__(self, d_model, pad_id=0, eps=1e-6):
        super(custom_ln, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(d_model))
        self.b_2 = torch.nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.pad_id = pad_id

    def forward(self, x):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / torch.sqrt(torch.var(x, dim=-1, keepdim=True) + self.eps)
        return self.a_2 * x + self.b_2
