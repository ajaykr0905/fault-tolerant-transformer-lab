from __future__ import annotations

import argparse
import json
from pathlib import Path

from fttl.dataset import PINNED_PEP_SOURCE, prepare_peps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and deterministically prepare the pinned Common Pile PEP corpus."
    )
    parser.add_argument("--cache-dir", type=Path, required=True, help="Gitignored download cache directory")
    parser.add_argument("--output", type=Path, required=True, help="Prepared dataset directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = prepare_peps(arguments.cache_dir, arguments.output)
    print(
        json.dumps(
            {
                "dataset": PINNED_PEP_SOURCE.repository,
                "revision": PINNED_PEP_SOURCE.revision,
                "documents": manifest.counts["documents"],
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "manifest": str(arguments.output / "manifest.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
