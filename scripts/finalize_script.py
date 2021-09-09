import argparse
from pathlib import Path


def finalize(path: Path):
    with open(path) as f:
        for line in f:  # noqa: B007
            pass
    last_line = line.replace("'", "").replace("\\", "").strip()
    key, value = last_line.split("=")
    assert key == "setup.viz.params"
    params = value.replace("[", "").replace("]", "")
    param_list = params.split(",")
    param_str = [f"{param}-${{{param}}}" for param in param_list]
    line_to_write = f"\n'setup.description={'----'.join(param_str)}'"
    with open(path, "a") as f:
        f.write(line_to_write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "script_id", metavar="N", type=str, help="script id to finalize"
    )
    args = parser.parse_args()
    path = Path(
        f"/private/home/sodhani/projects/abstraction_by_gating/scripts/{args.script_id}.sh"
    )
    finalize(path)
