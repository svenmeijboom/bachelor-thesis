from pathlib import Path
from typing import Optional, Union

from callbacks.base import BaseCallback

PathLike = Union[str, Path]


class SavePredictions(BaseCallback):
    def __init__(self, instances_dir: Optional[PathLike] = None, documents_dir: Optional[PathLike] = None):
        self.instances_dir = Path(instances_dir) if instances_dir is not None else None
        self.documents_dir = Path(documents_dir) if documents_dir is not None else None

    def on_evaluation_end(self, results: dict):
        for split_name, split_data in results.items():
            if 'instances' in split_data and self.instances_dir is not None:
                self.instances_dir.mkdir(exist_ok=True, parents=True)
                split_data['instances'].to_csv(self.instances_dir / f'instances-{split_name}.csv', index=False)

            if 'documents' in split_data and self.documents_dir is not None:
                self.documents_dir.mkdir(exist_ok=True, parents=True)
                split_data['documents'].to_csv(self.documents_dir / f'documents-{split_name}.csv', index=False)
