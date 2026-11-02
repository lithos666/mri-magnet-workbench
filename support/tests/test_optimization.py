"""Optimization correctness tests using an independent inexpensive toy model."""

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from mri_magnet.optimization import pareto_front, run_optimization


@dataclass(frozen=True)
class ToyDesign:
    radius_mm: float
    direction: int
    name: str = "baseline"

    def to_dict(self):
        return {"radius_mm": self.radius_mm, "direction": self.direction, "name": self.name}


def toy_metrics(design):
    # The two directions produce distinct strength/uniformity tradeoffs.
    field = 0.04 + 0.0001 * design.radius_mm + 0.01 * design.direction
    ppm = 50.0 + 3.0 * (design.radius_mm - 125.0) ** 2 + 90.0 * design.direction
    return {
        "mean_norm_T": field,
        "pp_norm_ppm": ppm,
        "rms_norm_ppm": ppm / 3.0,
        "mean_Bz_T": field,
        "pp_Bz_ppm": ppm,
        "center_Bz_T": field,
        "max_transverse_T": 0.0,
        "magnet_count": 28,
        "mass_magnet_kg": 1.0,
    }


class OptimizationTests(unittest.TestCase):
    def test_pareto_excludes_infeasible_and_dominated_designs(self):
        def record(name, field, ppm, feasible=True):
            return {
                "candidate_id": name,
                "metrics": {"mean_norm_T": field, "pp_norm_ppm": ppm},
                "feasible": feasible,
            }

        records = [
            record("uniform", 0.05, 20),
            record("strong", 0.10, 100),
            record("dominated", 0.04, 110),
            record("impossible", 5.0, 0, False),
        ]
        self.assertEqual({r["candidate_id"] for r in pareto_front(records)}, {"uniform", "strong"})

    def test_mixed_search_obeys_constraints_and_hard_budget_and_archives(self):
        calls = []

        def build(vector):
            self.assertEqual(float(vector[1]), int(vector[1]))
            return ToyDesign(float(vector[0]), int(vector[1]))

        def validate(design):
            valid = 120 <= design.radius_mm <= 130 and design.direction in (0, 1)
            return {"valid": valid, "violations": [] if valid else ["out of geometric envelope"]}

        def evaluate(design):
            self.assertTrue(validate(design)["valid"])
            calls.append(design)
            return toy_metrics(design)

        variables = [
            {"name": "radius_mm", "lower": 115, "upper": 135, "initial": 125, "integer": False},
            {"name": "direction", "lower": 0, "upper": 1, "initial": 0, "integer": True},
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = run_optimization(
                variables, build, evaluate, validate, ToyDesign(125, 0), folder,
                max_evaluations=22, weights=(0.0, 0.5, 1.0), popsize=4, maxiter=8, seed=12,
            )
            self.assertLessEqual(len(calls), 22)
            self.assertEqual(len(calls), result["metadata"]["field_evaluations"])
            self.assertEqual(result["baseline"]["candidate_id"], "baseline")
            self.assertIn(result["baseline"], result["records"])
            self.assertGreater(result["metadata"]["infeasible_proposals"], 0)
            self.assertEqual(len(result["metadata"]["rounds"]), 1)
            self.assertTrue(all(r["field_evaluations"] > 0 for r in result["metadata"]["rounds"]))
            self.assertTrue(all(validate(ToyDesign(**record["design"]))["valid"] for record in result["records"]))
            self.assertTrue(all(record["feasible"] for record in result["pareto"]))
            self.assertFalse(result["metadata"]["global_optimum_proven"])
            archived = [json.loads(line) for line in (Path(folder) / "optimization_records.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(archived, result["records"])
            saved = json.loads((Path(folder) / "optimization_result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["metadata"]["field_evaluations"], len(calls))
            self.assertTrue((Path(folder) / "optimization_iterations.jsonl").read_text(encoding="utf-8"))

    def test_geometry_aliases_share_evaluations_across_weights(self):
        calls = []

        def build(vector):
            return ToyDesign(125, int(vector[0]) % 2, "renamed candidate")

        def evaluate(design):
            calls.append(design)
            return toy_metrics(design)

        with tempfile.TemporaryDirectory() as folder:
            result = run_optimization(
                [{"name": "direction_alias", "lower": 0, "upper": 5, "initial": 0, "integer": True}],
                build, evaluate, lambda design: {"valid": True, "violations": []},
                ToyDesign(125, 0), folder, max_evaluations=20,
                weights=(0, 0.5, 1), popsize=6, maxiter=3, seed=9,
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result["records"]), 2)
        self.assertGreater(result["metadata"]["cache_hits"], 0)
        self.assertTrue(result["metadata"]["baseline_matches_initial_vector"])
        self.assertEqual(len(result["pareto"]), 2)

    def test_baseline_only_budget_returns_reviewable_partial_result(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_optimization(
                [{"name": "radius_mm", "lower": 120, "upper": 130, "initial": 125, "integer": False}],
                lambda vector: ToyDesign(float(vector[0]), 0), toy_metrics,
                lambda design: {"valid": True, "violations": []}, ToyDesign(125, 0), folder,
                max_evaluations=1,
            )
        self.assertEqual(result["state"], "budget_exhausted")
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["metadata"]["field_evaluations"], 1)
        self.assertEqual(result["pareto"], [result["baseline"]])

    def test_invalid_baseline_never_reaches_field_solver(self):
        def forbidden_evaluation(design):
            self.fail("An invalid baseline reached field evaluation")

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "baseline design violates"):
                run_optimization(
                    [], lambda vector: ToyDesign(100, 0), forbidden_evaluation,
                    lambda design: {"valid": False, "violations": ["bore intrusion"]},
                    ToyDesign(100, 0), folder,
                )


if __name__ == "__main__":
    unittest.main()
