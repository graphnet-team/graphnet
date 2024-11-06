"""`Sampler` and `BatchSampler` objects for `graphnet`.

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
"""

from typing import (
    Any,
    List,
    Optional,
    Tuple,
    Iterator,
    Sequence,
)

from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import torch
from torch.utils.data import Sampler, BatchSampler
from graphnet.data.dataset import Dataset
from graphnet.utilities.logging import Logger


class RandomChunkSampler(Sampler[int]):
    """A `Sampler` that randomly selects chunks.

    Original implementation:
    https://github.com/DrHB/icecube-2nd-place/blob/main/src/dataset.py
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


def gather_len_matched_buckets(
    params: Tuple[range, Sequence[Any], int, int],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Gather length-matched buckets of events.

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


class LenMatchBatchSampler(BatchSampler, Logger):
    """A `BatchSampler` that batches similar length events.

    Original implementation:
    https://github.com/DrHB/icecube-2nd-place/blob/main/src/dataset.py
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int = 1,
        num_workers: int = 1,
        bucket_width: int = 16,
        chunks_per_segment: int = 4,
        multiprocessing_context: str = "spawn",
        drop_last: Optional[bool] = False,
    ) -> None:
        """Construct `LenMatchBatchSampler`.

        This `BatchSampler` groups data with similar lengths to be more
        efficient in operations like masking for MultiHeadAttention. Since
        batch samplers run on the main process and can result in a CPU
        bottleneck, `num_workers` can be specified to use multiprocessing for
        creating the batches. The `bucket_width` argument specifies how wide
        the bins are for grouping batches. For example, with `bucket_width=16`,
        data with length [1, 16] are grouped into a bucket, data with length
        [17, 32] into another, etc.

        Args:
            sampler: A `Sampler` object that selects/draws data in some way.
            batch_size: Batch size.
            num_workers: Number of workers to spawn to create batches.
            bucket_width: Size of length buckets for grouping data.
            chunks_per_segment: Number of chunks to group together.
            multiprocessing_context: Start method for multiprocessing.
            drop_last: (Optional) Drop the last incomplete batch.
        """
        Logger.__init__(self)
        super().__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )
        assert num_workers >= 0, "`num_workers` must be >= 0!"

        self._num_workers = num_workers
        self._bucket_width = bucket_width
        self._chunks_per_segment = chunks_per_segment
        self._multiprocessing_context = multiprocessing_context

        self.info(
            f"Setting up batch sampler with {self._num_workers} workers."
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Return length-matched batches."""
        indices = list(self.sampler)
        data_source = self.sampler.data_source

        if self._num_workers > 0:

            n_chunks = len(self.sampler.chunks)
            n_segments = n_chunks // self._chunks_per_segment

            # Split indices into nearly equal-sized segments amongst workers
            segments = [
                range(
                    sum(self.sampler.chunks[: i * self._chunks_per_segment]),
                    sum(
                        self.sampler.chunks[
                            : (i + 1) * self._chunks_per_segment
                        ]
                    ),
                )
                for i in range(n_segments)
            ]
            segments.extend(
                [range(segments[-1][-1], len(indices) - 1)]
            )  # Make a segment w/ the leftover indices

            remaining_indices = []
            with get_context(self._multiprocessing_context).Pool(
                processes=self._num_workers
            ) as pool:
                results = pool.imap_unordered(
                    gather_len_matched_buckets,
                    [
                        (
                            segments[i],
                            data_source,
                            self.batch_size,
                            self._bucket_width,
                        )
                        for i in range(n_segments)
                    ],
                )
                for result in results:
                    batches, leftovers = result
                    for batch in batches:
                        yield batch
                    remaining_indices.extend(leftovers)

            # Process any remaining indices
            batch = []
            for incomplete_batch in remaining_indices:
                batch.extend(incomplete_batch)
                if len(batch) >= self.batch_size:
                    yield batch[: self.batch_size]
                    batch = batch[self.batch_size :]

            if len(batch) > 0 and not self.drop_last:
                yield batch
        else:  # n_workers = 0, no multiprocessing
            buckets = defaultdict(list)

            for idx in self.sampler:
                s = self.sampler.data_source[idx]
                L = max(1, s.num_nodes // self._bucket_width)
                buckets[L].append(idx)
                if len(buckets[L]) == self.batch_size:
                    batch = list(buckets[L])
                    yield batch
                    buckets[L] = []

            batch = []
            leftover = [idx for bucket in buckets for idx in bucket]

            for idx in leftover:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            if len(batch) > 0 and not self.drop_last:
                yield batch
