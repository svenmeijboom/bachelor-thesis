from collections import defaultdict
import random
from typing import List, Iterator, Optional, Tuple

from torch.utils.data import Sampler, BatchSampler

from information_extraction.training.data.base import BaseDataset


class DocumentSampler(Sampler[List[int]]):
    """
    Samples from the dataset on a per-document basis. Ensures the segments for each document are split into
    batches of a specified batch size, and that all segments in a batch belong to the same document and attribute.
    """
    def __init__(self, data_source: BaseDataset, num_documents: Optional[int] = None, batch_size: int = 1,
                 shuffle: bool = False, replacement: bool = False, remove_null: bool = False, generator=None) -> None:
        self.data_source = data_source
        self.num_documents = num_documents
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replacement = replacement
        self.remove_null = remove_null
        self.generator = generator

        self.document_indices = defaultdict(lambda: defaultdict(list))
        indices = self.data_source.indices_with_data if remove_null else range(len(self.data_source))

        for i in indices:
            self.document_indices[self.data_source.docs[i]][self.data_source.features[i]].append(i)

    def __iter__(self) -> Iterator[List[int]]:
        documents = list(self.document_indices)
        num_returned = 0

        while True:
            if self.shuffle:
                random.shuffle(documents)

            for doc_id in documents:
                if self.num_documents is not None and num_returned >= self.num_documents:
                    return

                for attribute_indices in self.document_indices[doc_id].values():
                    yield from BatchSampler(attribute_indices, self.batch_size, drop_last=False)

                num_returned += 1

            if not self.replacement:
                return
