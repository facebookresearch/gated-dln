from __future__ import annotations

import json
import sys
from typing import Any

sys.path.append("/private/home/sodhani/projects/abstraction_by_gating/")


def update_path(current_path: str) -> str:
    return current_path.replace("/data/home", "/private/home").replace(
        "abstraction-by-gating", "abstraction_by_gating"
    )


def get_unique_records_from_file_dump(
    file_path: str = "/private/home/sodhani/projects/abstraction_by_gating/job.log",
) -> dict[str, [dict[str, Any]]]:

    dedup_data = {}
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            current_path = data["logbook"]["logger_dir"]
            if current_path.startswith("/data/home"):
                data["logbook"]["logger_dir"] = update_path(current_path)
            data["setup"]["base_path"] = update_path(data["setup"]["base_path"])
            data["setup"]["save_dir"] = update_path(data["setup"]["save_dir"])
            data["dataloader"]["train_config"]["dataset"]["root"] = update_path(
                data["dataloader"]["train_config"]["dataset"]["root"]
            )
            data["dataloader"]["test_config"]["dataset"]["root"] = data["dataloader"][
                "test_config"
            ]["dataset"]["root"]
            key = data["setup"]["id"]

            dedup_data[key] = data

    return dedup_data
