"""`Sampler` and `BatchSampler` objects for `graphnet`."""
from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Iterator,
    Sequence,
)

from collections import defaultdict
from multiprocessing import Pool, cpu_count, get_context

import numpy as np
import torch
from torch.utils.data import Sampler, BatchSampler
from graphnet.data.dataset import Dataset


class RandomChunkSampler(Sampler[int]):
    """A `Sampler` that randomly selects chunks.

    MIT License

    Copyright (c) 2023 DrHB

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    Original implementation: https://github.com/DrHB/icecube-2nd-place/blob/main/src/dataset.py
    """

    def __init__(
        self,
        data_source: Dataset,
        num_samples: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Construct `RandomChunkSampler`."""
        self._data_source = data_source
        self._num_samples = num_samples
        self._chunks = data_source.chunk_sizes

        # Create a random number generator if one was not provided
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)
        else:
            self._generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def data_source(self) -> Sequence[Any]:
        """Return the data source."""
        return self._data_source

    @property
    def num_samples(self) -> int:
        """Return the number of samples in the data source."""
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __len__(self) -> int:
        """Return the number of sampled."""
        return self.num_samples

    @property
    def chunks(self) -> List[int]:
        """Return the list of chunks."""
        return self._chunks

    def __iter__(self) -> Iterator[List[int]]:
        """Return a list of indices from a randomly sampled chunk."""
        cumsum = np.cumsum(self.chunks)
        chunk_list = torch.randperm(
            len(self.chunks), generator=self._generator
        ).tolist()

        # sample indexes chunk by chunk
        yield_samples = 0
        for i in chunk_list:
            chunk_len = self.chunks[i]
            offset = cumsum[i - 1] if i > 0 else 0
            samples = (
                offset + torch.randperm(chunk_len, generator=self._generator)
            ).tolist()
            if len(samples) <= self.num_samples - yield_samples:
                yield_samples += len(samples)
            else:
                samples = samples[: self.num_samples - yield_samples]
                yield_samples = self.num_samples
            yield from samples


def gather_buckets(
    params: Tuple[List[int], Sequence[Any], int, int],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Gather buckets of events.

    The function that will be used to gather batches of events for the
    `LenMatchBatchSampler`. When using multiprocessing, each worker will call
    this function. Given indices, this function will group events based on
    their length. If the length of event is N, then it will go into the
    (N // bucket_width) bucket. This returns completed batches and a
    list of incomplete batches that did not fill to batch_size at the end.

    Args:
        params: A tuple containg the list of indices to process,
        the data_source (typically a `Dataset`), the batch size, and the
        bucket width.

    Returns:
        batches: A list containing batches.
        remaining_batches: Incomplete batches.
    """
    indices, data_source, batch_size, bucket_width = params
    buckets = defaultdict(list)
    batches = []

    for idx in indices:
        s = data_source[idx]
        L = max(1, s.num_nodes // bucket_width)
        buckets[L].append(idx)
        if len(buckets[L]) == batch_size:
            batches.append(list(buckets[L]))
            buckets[L] = []

    # Include any remaining items in partially filled buckets
    remaining_batches = [b for b in buckets.values() if b]
    return batches, remaining_batches


class LenMatchBatchSampler(BatchSampler):
    """A `BatchSampler` that batches similar length events.

    MIT License

    Copyright (c) 2023 DrHB

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    Original implementation: https://github.com/DrHB/icecube-2nd-place/blob/main/src/dataset.py
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int = 1,
        num_workers: int = 1,
        bucket_width: int = 16,
        drop_last: Optional[bool] = False,
    ) -> None:
        """Construct `LenMatchBatchSampler`."""
        super().__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )
        self._bucket_width = bucket_width
        self._num_workers = num_workers

    def __iter__(self) -> Iterator[List[int]]:
        """Return length-matched batches."""
        indices = list(self.sampler)
        data_source = self.sampler.data_source

        chunk_size = len(indices) // self._num_workers

        # Split indices into nearly equal-sized chunks
        chunks = [
            indices[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self._num_workers)
        ]
        if len(indices) % self._num_workers != 0:
            chunks.append(indices[self._num_workers * chunk_size :])

        yielded = 0
        with get_context("spawn").Pool(processes=self._num_workers) as pool:
            results = pool.map(
                gather_buckets,
                [
                    (chunk, data_source, self.batch_size, self._bucket_width)
                    for chunk in chunks
                ],
            )

        merged_batches = []
        remaining_indices = []
        for batches, remaining in results:
            merged_batches.extend(batches)
            remaining_indices.extend(remaining)

        for batch in merged_batches:
            yield batch
            yielded += 1

        # Process any remaining indices
        leftover = [idx for batch in remaining_indices for idx in batch]
        batch = []
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                yielded += 1
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch
            yielded += 1
