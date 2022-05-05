import os
from pathlib import Path

import wandb

from transformers import T5Tokenizer

from feature_extraction.text import TextExtractor


def main():
    wandb.init(
        project='information_extraction',
        name='text-feature-extraction',
        group='feature-extraction',
        job_type='preprocess',
    )

    input_artifact = wandb.use_artifact('train-val-test-split:latest')
    input_dir = Path(input_artifact.download(root=os.path.expanduser('~/Data/SWDE-split/')))
    output_dir = Path('~/Data/SWDE-text/').expanduser()

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    for context_size in [128, 256, 512]:
        extractor = TextExtractor(input_dir, output_dir, tokenizer, context_size,
                                  num_workers=48)
        extractor.process_dataset()

    artifact = wandb.Artifact('swde-text', type='preprocessed-data',
                              description='Preprocessed SWDE dataset where contexts are represented as pure text')
    artifact.add_dir(str(output_dir))

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    main()
