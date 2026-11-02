"""Sample an existing MPH solution without running its study or saving it."""
from pathlib import Path
import argparse
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mri_magnet.comsol_bridge import prepare_existing_solution_export, run_existing_solution_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mph", required=True)
    parser.add_argument("--comsol-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--points-csv", help="CSV with x_mm,y_mm,z_mm columns")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.points_csv:
        with Path(args.points_csv).open(encoding="utf-8-sig", newline="") as stream:
            points = [[float(row[c]) for c in ("x_mm", "y_mm", "z_mm")] for row in csv.DictReader(stream)]
    else:
        points = [[0, 0, 0]]
        for radius in (10, 30, 90):
            for axis in range(3):
                for sign in (-1, 1):
                    point = [0, 0, 0]; point[axis] = sign * radius; points.append(point)
    job = prepare_existing_solution_export(args.mph, points, args.output, args.comsol_root)
    print(json.dumps({"prepared": job["java_path"], "points": len(points)}, ensure_ascii=False), flush=True)
    if not args.prepare_only:
        print(json.dumps(run_existing_solution_export(job, timeout=args.timeout), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
