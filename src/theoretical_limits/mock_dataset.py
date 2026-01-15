import time

import torch
from torch.utils.data import IterableDataset


class MockDataset(IterableDataset):

    def __init__(self, batch_size, sleep_time, input_size, output_size, n_batches):
        self.sleep_time = sleep_time
        self.n_batches = n_batches

        self.x_mock = torch.ones((batch_size, input_size), dtype=torch.float32)
        self.y_mock = torch.ones(batch_size, dtype=torch.int64)

    def __iter__(self):
        for _ in range(self.n_batches):
            time.sleep(self.sleep_time)
            yield self.x_mock, self.y_mock