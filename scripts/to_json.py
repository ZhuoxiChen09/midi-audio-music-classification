"""Convert literal-format submission predictions to a separate JSON file."""

import argparse
import ast
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    predictions = ast.literal_eval(args.source.read_text(encoding="utf-8"))
    if not isinstance(predictions, dict):
        parser.error("predictions must be a mapping")
    content = json.dumps(predictions, allow_nan=False)
    try:
        with args.destination.open("x", encoding="utf-8") as file:
            file.write(content + "\n")
    except FileExistsError:
        parser.error(f"destination already exists: {args.destination}")


if __name__ == "__main__":
    main()
