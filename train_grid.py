import argparse
from pathlib import Path

import flair
import torch
from flair.data import Corpus

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.embeddings import (
    StackedEmbeddings,
    FlairEmbeddings,
    TokenEmbeddings,
)

from flair.datasets import UniversalDependenciesCorpus
from typing import Union
from flair.file_utils import cached_path


class UD_UKRAINIAN(UniversalDependenciesCorpus):
    def __init__(self, base_path: Union[str, Path] = None, in_memory: bool = True, split_multiwords: bool = True):
        if type(base_path) == str:
            base_path = Path(base_path)

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = flair.cache_root / "datasets"
        data_folder = base_path / dataset_name

        # download data if necessary
        ud_path = "https://raw.githubusercontent.com/UniversalDependencies/UD_Ukrainian-IU/master"
        cached_path(f"{ud_path}/uk_iu-ud-dev.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/uk_iu-ud-test.conllu", Path("datasets") / dataset_name)
        cached_path(f"{ud_path}/uk_iu-ud-train.conllu", Path("datasets") / dataset_name)

        super(UD_UKRAINIAN, self).__init__(data_folder, in_memory=in_memory, split_multiwords=split_multiwords)


def choochoochoo(embeddings: TokenEmbeddings) -> None:

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = UD_UKRAINIAN()

    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings()])

    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[64, 128, 256])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.25])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[8, 16, 32])

    param_selector = SequenceTaggerParamSelector(
        corpus, "upos", base_path=Path("./pos-tests/flair.grid/"), training_runs=3, max_epochs=150
    )

    # start the optimization
    param_selector.optimize(search_space, max_evals=100)


if __name__ == "__main__":
    flair.device = torch.device("cuda:0")

    parser = argparse.ArgumentParser(
        description="""That is the hyperparam opt trainer that can accept a base dir with embeddings"""
    )

    parser.add_argument("--embeddings-dir", type=Path, help="Path base dir with embeddings", default=Path("/data/"))

    args = parser.parse_args()

    choochoochoo(
        lambda: StackedEmbeddings(
            [
                FlairEmbeddings(args.embeddings_dir / "flair/uk/backward/best-lm.pt"),
                FlairEmbeddings(args.embeddings_dir / "flair/uk/forward/best-lm.pt"),
            ]
        )
    )
