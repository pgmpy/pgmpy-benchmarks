"""Compare two commits' results with asv's own `compare` logic, with a usable exit status.

`asv compare` always exits 0, and `asv continuous` skips the comparison when any
benchmark errored. This runs the same table/regression logic and exits 1 when a
benchmark got `factor` times slower or newly fails on head. Benchmarks that error
on both commits are neutral.
"""

import argparse

from asv import results as asv_results
from asv import util as asv_util
from asv.commands.compare import Compare
from asv.config import Config


class _Compare(Compare):
    worsened = False
    improved = False

    @classmethod
    def print_table(cls, *args, **kwargs):
        worsened, improved = super().print_table(*args, **kwargs)
        cls.worsened, cls.improved = worsened, improved
        return worsened, improved


def _has_results(results_dir, machine, commit):
    try:
        return bool(list(asv_results.iter_results_for_machine_and_hash(results_dir, machine, commit)))
    except asv_util.UserError:  # unknown or ambiguous commit prefix
        return False


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--head", required=True)
    p.add_argument("--factor", type=float, default=1.5)
    p.add_argument("--machine", default=None)
    p.add_argument("--config", default="asv.conf.json")
    args = p.parse_args(argv)

    conf = Config.load(args.config)

    missing = [
        c for c in (args.base, args.head) if not _has_results(conf.results_dir, args.machine, c)
    ]
    for commit in missing:
        print(f"::error title=No benchmark results::{commit}")
    if missing:
        return 1

    _Compare.worsened = _Compare.improved = False
    _Compare.run(
        conf,
        hash_1=args.base,
        hash_2=args.head,
        factor=args.factor,
        machine=args.machine,
        split=True,
        only_changed=False,
        use_stats=True,
    )
    return 1 if _Compare.worsened else 0


if __name__ == "__main__":
    raise SystemExit(main())
