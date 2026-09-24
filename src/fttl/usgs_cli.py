from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fttl.usgs import (
    USGS_FEEDS,
    USGSCaptureLedger,
    fetch_feed,
    prepare_snapshot_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture versioned USGS earthquake events and seal a deterministic offline snapshot. "
            "This is ingestion, not online learning or earthquake prediction."
        )
    )
    parser.add_argument("--feed", choices=sorted(USGS_FEEDS), default="all-day")
    parser.add_argument("--database", type=Path, default=Path(".cache/fttl/usgs.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path(".cache/fttl/usgs-v1"))
    parser.add_argument("--polls", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if not 1 <= args.polls <= 60:
        parser.error("--polls must be between 1 and 60")
    if not 1 <= args.interval_seconds <= 3600:
        parser.error("--interval-seconds must be between 1 and 3600")

    feed_url = USGS_FEEDS[args.feed]
    captures = []
    with USGSCaptureLedger(args.database) as ledger:
        for poll_index in range(args.polls):
            payload = fetch_feed(feed_url, timeout_seconds=args.timeout_seconds)
            captures.append(
                ledger.capture(
                    payload,
                    feed=args.feed,
                    captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                )
            )
            if poll_index + 1 < args.polls:
                time.sleep(args.interval_seconds)
        data_path, manifest_path, manifest = ledger.seal_snapshot(
            args.output,
            feed=args.feed,
        )
    training_manifest = prepare_snapshot_dataset(
        manifest_path,
        args.output / "training",
    )
    print(
        json.dumps(
            {
                "captures": [capture.__dict__ for capture in captures],
                "data_path": str(data_path),
                "manifest": manifest.to_dict(),
                "manifest_path": str(manifest_path),
                "training_dataset_fingerprint": training_manifest.dataset_fingerprint,
                "training_manifest_path": str(args.output / "training" / "manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
