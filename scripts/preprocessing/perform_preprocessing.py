from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Optional

import wandb

from transformers import T5Tokenizer

from information_extraction.preprocessing.feature_extraction.text import TextExtractor
from information_extraction.preprocessing.feature_extraction.html import HtmlExtractor
from information_extraction.config import WANDB_PROJECT, DATA_DIR


NUM_WORKERS = 48


def main(representation: str, parent_depth: Optional[int] = None):
    wandb.init(
        project=WANDB_PROJECT,
        name=f'{representation}-feature-extraction',
        group='feature-extraction',
        job_type='preprocess',
    )

    input_dir = DATA_DIR / 'SWDE-split'
    output_dir = DATA_DIR / 'SWDE-html'

    input_artifact = wandb.use_artifact('random-split:latest')
    input_artifact.download(root=str(input_dir))

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    for context_size in [128, 256, 512]:
        if representation == 'html':
            extractor = HtmlExtractor(input_dir, output_dir, tokenizer, context_size,
                                      parent_depth=parent_depth, encode_id=True, encode_class=True,
                                      encode_tag_subset={'div', 'a', 'span', 'p'}, num_workers=NUM_WORKERS)
            metadata = {'Parent depth': parent_depth}
        else:
            extractor = TextExtractor(input_dir, output_dir, tokenizer, context_size,
                                      num_workers=NUM_WORKERS)
            metadata = None

        extractor.process_dataset()

    descriptions = {
        'text': 'Preprocessed SWDE dataset where contexts are represented as pure text',
        'html': 'Preprocessed SWDE dataset where HTML structure is included in the context',
    }

    artifact = wandb.Artifact(f'swde-{representation}', type='preprocessed-data', metadata=metadata,
                              description=descriptions[representation])
    artifact.add_dir(str(output_dir))

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess the SWDE dataset into segments and sequence representations')

    parser.add_argument('-r', '--representation', choices=['text', 'html'], required=True,
                        help='Which representation to use')
    parser.add_argument('-p', '--parent-depth', type=int,
                        help='The amount of ancestors to encode in the HTML representation')

    args = parser.parse_args()

    if args.representation == 'html' and args.parent_depth is None:
        raise ValueError('Parent depth must be supplied when representation is HTML')

    main(args.representation, args.parent_depth)
