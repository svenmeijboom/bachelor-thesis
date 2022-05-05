import os
from pathlib import Path

import wandb

from transformers import T5Tokenizer

from feature_extraction.html import HtmlExtractor


PARENT_DEPTH = 3


def main():
    wandb.init(
        project='information_extraction',
        name='html-feature-extraction',
        group='feature-extraction',
        job_type='preprocess',
    )

    input_artifact = wandb.use_artifact('train-val-test-split:latest')
    input_dir = Path(input_artifact.download(root=os.path.expanduser('~/Data/SWDE-split/')))
    output_dir = Path('~/Data/SWDE-html/').expanduser()

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    for context_size in [128, 256, 512]:
        extractor = HtmlExtractor(input_dir, output_dir, tokenizer, context_size,
                                  parent_depth=PARENT_DEPTH, num_workers=48)
        extractor.process_dataset()

    artifact = wandb.Artifact('swde-html', type='preprocessed-data',
                              description='Preprocessed SWDE dataset where HTML structure is included in the context',
                              metadata={'Parent depth': PARENT_DEPTH})
    artifact.add_dir(str(output_dir))

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    main()
