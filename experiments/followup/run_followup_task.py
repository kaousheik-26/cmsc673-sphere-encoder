"""Execute one explicit comparison entry from a saved Slurm array plan."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    task = json.loads(args.plan.read_text())[args.index]
    command = [sys.executable, "-u", "-m", "experiments.followup.compare_adaptive"]
    for key, value in task.items():
        if isinstance(value, list):
            command.extend(["--" + key, *map(str, value)])
        else:
            command.extend(["--" + key, str(value)])
    print(json.dumps({"plan": str(args.plan), "index": args.index, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
