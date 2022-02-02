from __future__ import annotations

import json
import sys
from typing import Any

sys.path.append("/private/home/sodhani/projects/abstraction_by_gating/")


def get_unique_records_from_file_dump(
    file_path: str = "/private/home/sodhani/projects/abstraction_by_gating/job.log",
) -> dict[str, [dict[str, Any]]]:

    dedup_data = {}
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            if data["logbook"]["logger_dir"].startswith("/data/home"):
                data["logbook"]["logger_dir"] = (
                    data["logbook"]["logger_dir"]
                    .replace("/data/home", "/private/home")
                    .replace("abstraction-by-gating", "abstraction_by_gating")
                )
            key = data["setup"]["id"]

            dedup_data[key] = data

    return dedup_data
