"""Benchmarks for ``pgmpy.registry``. Calls are looped ``CALLS`` times: a single call takes ~1 us, near the clock quantum."""

from benchmarks._bench import Bench

from pgmpy.registry import all_tags, check_tag_is_valid


CALLS = 100


class TimeRegistry(Bench):
    def time_all_tags(self):
        for _ in range(CALLS):
            all_tags()

    def time_all_tags_one_type(self):
        for _ in range(CALLS):
            all_tags("ci_test")

    def time_all_tags_dataframe(self):
        all_tags(as_dataframe=True)

    def time_check_tag_is_valid_list(self):
        for _ in range(CALLS):
            check_tag_is_valid("data_types", ["discrete", "continuous"])

    def time_check_tag_is_valid_bool(self):
        for _ in range(CALLS):
            check_tag_is_valid("requires_data", True)
