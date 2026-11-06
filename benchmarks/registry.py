"""Benchmarks for ``pgmpy.registry``.

``all_tags`` and ``check_tag_is_valid`` are the module's public functions (``OBJECT_TYPES`` and
``TAG_REGISTER`` are data). They work on a fixed register of a few dozen tags, so they have no size
and only a ``fast`` tier, like the listings in ``datasets.py``. ``all_tags`` is timed for every object
type, for one object type, and as a data frame; ``check_tag_is_valid`` for a list-valued tag
(``data_types``) and for a bool tag (``requires_data``), each with a valid value.

All but the data frame variant take 1 to 2 microseconds, which is about the clock quantum on some
machines, so each timed call makes ``CALLS`` calls and the noise check can tell 1.5x from 1x.
"""

from benchmarks._tiers import Bench

from pgmpy.registry import all_tags, check_tag_is_valid


CALLS = 100


class TimeRegistry(Bench):
    params = [["fast"]]

    def time_all_tags(self, tier):
        for _ in range(CALLS):
            all_tags()

    def time_all_tags_one_type(self, tier):
        for _ in range(CALLS):
            all_tags("ci_test")

    def time_all_tags_dataframe(self, tier):
        all_tags(as_dataframe=True)

    def time_check_tag_is_valid_list(self, tier):
        for _ in range(CALLS):
            check_tag_is_valid("data_types", ["discrete", "continuous"])

    def time_check_tag_is_valid_bool(self, tier):
        for _ in range(CALLS):
            check_tag_is_valid("requires_data", True)
