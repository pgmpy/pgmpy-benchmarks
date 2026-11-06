"""Benchmarks for the model and dataset loaders (``pgmpy.example_models`` and ``pgmpy.datasets``).

``get_example_model`` of ``pgmpy.utils`` is deprecated in favour of ``pgmpy.example_models.load_model``,
so the loaders are ``load_model`` / ``list_models`` and ``load_dataset`` / ``list_datasets``. The models
and datasets live on the Hugging Face Hub and are cached on disk after the first download.

``TimeLoaders`` times ``load_model`` (``bnlearn/asia`` fast, ``bnlearn/munin`` slow) and
``load_dataset`` (``DATASET``, ``rng=0``). The inputs are hard-coded, not picked at run time, so base
and head load the same data. Offline timing: ``prepare`` loads the input once per process (the
download and cache warm-up, memoized by ``warmed``), then sets ``huggingface_hub.constants.HF_HUB_OFFLINE``
to ``True``, and ``teardown`` restores the value found at import. ``huggingface_hub`` reads that constant
at call time (``is_offline_mode()``; ``hf_hub_download`` then skips the revision check and reads the
cache), so the timed call does no network I/O and measures parsing only. ``tests/test_suite.py``
(``test_loader_timed_call_is_offline``) pins that.

Dataset choice (ruling R25), measured with ``PYTHONHASHSEED=0`` on 52 entries of ``list_datasets()``,
sized by rows x columns after loading. The fast entry is the smallest:

* ``iq_brain_size``: 20 x 9 = 180 cells; first load 0.17 s, offline load under 10 ms.

The slow entry is the largest whose first load takes under 60 s and whose offline load fits the slow
budget (about 1 s on average):

* ``cover_type``: 581012 x 11 = 6.4 M cells; first load 3.5 s, offline load 0.06 s.
* Excluded, larger: ``causalbench_rpe1`` (162733 x 383 = 62 M cells; first load 8.9 s, but 2.2 s offline)
  and ``causalbench_k562`` (162751 x 622, larger still); both exceed the budget offline.
* ``angrist_krueger_qob`` (329509 x 5) has no data file on the Hub (404) and cannot be loaded.
* ``tubingen`` is loaded as ``tubingen/<pair_id>`` and simulated entries (``anm``, ``linear_gaussian``)
  have no size tag, so they are not candidates.

Offline load of the models: ``bnlearn/asia`` 0.29 s, ``bnlearn/munin`` 3.0 s (BIF parsing plus the
registry lookup; munin is above the 1 s guideline but far below the 15 s cap, and the brief fixes it).

``TimeListings`` (fast only, no size) covers ``list_models`` and ``list_datasets``; both only scan the
registry and do no I/O.

Skipped:
    ``load_interventions``, ``load_expert_knowledge`` and ``load_ground_truth`` (the classmethods of the
    dataset classes, not exported functions; ``load_dataset`` runs them), ``get_example_model``
    (deprecated), ``BaseDataset`` and ``BaseExampleModel`` (abstract).
"""

from functools import cache

from benchmarks._tiers import Bench

from huggingface_hub import constants
from pgmpy.datasets import list_datasets, load_dataset
from pgmpy.example_models import list_models, load_model

MODEL = {"fast": "bnlearn/asia", "slow": "bnlearn/munin"}
DATASET = {"fast": "iq_brain_size", "slow": "cover_type"}

ORIGINAL_OFFLINE = constants.HF_HUB_OFFLINE


@cache
def warmed(tier):
    """Load the tier's model and dataset once so that the Hugging Face cache holds their files."""
    load_model(MODEL[tier])
    load_dataset(DATASET[tier], rng=0)


class TimeLoaders(Bench):
    def prepare(self, tier):
        warmed(tier)
        constants.HF_HUB_OFFLINE = True

    def teardown(self, tier):
        constants.HF_HUB_OFFLINE = ORIGINAL_OFFLINE

    def time_load_model(self, tier):
        load_model(MODEL[tier])

    def time_load_dataset(self, tier):
        load_dataset(DATASET[tier], rng=0)


class TimeListings(Bench):
    params = [["fast"]]

    def time_list_models(self, tier):
        list_models()

    def time_list_datasets(self, tier):
        list_datasets()
