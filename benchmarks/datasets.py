"""Benchmarks for the loaders in ``pgmpy.example_models`` and ``pgmpy.datasets``.

Loaders run offline: ``setup`` warms the Hugging Face cache once, then sets ``HF_HUB_OFFLINE``; ``teardown`` restores it
(pinned by ``test_loader_timed_call_is_offline``).

Skipped:
    load_interventions, load_expert_knowledge, load_ground_truth: not exported (run by ``load_dataset``)
    get_example_model: deprecated
    BaseDataset, BaseExampleModel: abstract
"""

from functools import cache

from benchmarks._bench import Bench

from huggingface_hub import constants
from pgmpy.datasets import list_datasets, load_dataset
from pgmpy.example_models import list_models, load_model

MODEL = "bnlearn/asia"
DATASET = "iq_brain_size"

ORIGINAL_OFFLINE = constants.HF_HUB_OFFLINE


@cache
def warmed():
    """Load the model and dataset once so that the Hugging Face cache holds their files."""
    load_model(MODEL)
    load_dataset(DATASET, rng=0)


class TimeLoaders(Bench):
    def setup(self):
        warmed()
        constants.HF_HUB_OFFLINE = True

    def teardown(self):
        constants.HF_HUB_OFFLINE = ORIGINAL_OFFLINE

    def time_load_model(self):
        load_model(MODEL)

    def time_load_dataset(self):
        load_dataset(DATASET, rng=0)


class TimeListings(Bench):
    def time_list_models(self):
        list_models()

    def time_list_datasets(self):
        list_datasets()
