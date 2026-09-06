"""Tests for the metrics, the model definition and the recorded predictions.

The point of storing predictions beside the metrics is that the metrics can be
recomputed from them without a GPU. These tests do that, so a number edited by
hand in the results file fails here rather than surviving to the README.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.benchmark import CLASSES, confusion_pairs, scores

RESULTS = Path(__file__).resolve().parents[1] / "results" / "benchmark.json"


class Scores(unittest.TestCase):
    def test_a_perfect_prediction_scores_one(self):
        actual = np.arange(len(CLASSES)).repeat(5)
        result = scores(actual, actual)
        self.assertEqual(result["top1_accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)

    def test_predicting_one_class_scores_the_base_rate(self):
        actual = np.arange(len(CLASSES)).repeat(5)
        result = scores(actual, np.zeros_like(actual))
        self.assertAlmostEqual(result["top1_accuracy"], 1 / len(CLASSES))
        self.assertLess(result["macro_f1"], 0.1)

    def test_accuracy_equals_macro_recall_on_a_balanced_split(self):
        """Why accuracy alone is not enough: it carries no precision information."""
        rng = np.random.default_rng(0)
        actual = np.arange(len(CLASSES)).repeat(40)
        predicted = np.where(rng.random(len(actual)) < 0.3,
                             rng.integers(0, len(CLASSES), len(actual)), actual)
        result = scores(actual, predicted)
        macro_recall = np.mean([entry["recall"]
                                for entry in result["per_class"].values()])
        self.assertAlmostEqual(result["top1_accuracy"], macro_recall, places=10)

    def test_every_class_is_reported_even_when_never_predicted(self):
        actual = np.arange(len(CLASSES)).repeat(3)
        result = scores(actual, np.zeros_like(actual))
        self.assertEqual(tuple(result["per_class"]), CLASSES)
        self.assertEqual(result["per_class"]["truck"]["precision"], 0.0)

    def test_the_confusion_matrix_rows_sum_to_the_support(self):
        rng = np.random.default_rng(1)
        actual = np.arange(len(CLASSES)).repeat(7)
        predicted = rng.integers(0, len(CLASSES), len(actual))
        result = scores(actual, predicted)
        for index, name in enumerate(CLASSES):
            self.assertEqual(sum(result["confusion_matrix"][index]),
                             result["per_class"][name]["support"])


class ConfusionPairs(unittest.TestCase):
    def test_pairs_exclude_the_diagonal_and_come_out_ordered(self):
        actual = np.array([0] * 20 + [1] * 20)
        predicted = np.array([1] * 6 + [0] * 14 + [0] * 9 + [1] * 11)
        matrix = scores(actual, predicted)["confusion_matrix"]
        pairs = confusion_pairs(matrix)
        self.assertEqual(pairs[0]["count"], 9)
        self.assertEqual(pairs[1]["count"], 6)
        for entry in pairs:
            self.assertNotEqual(entry["true"], entry["predicted"])

    def test_shares_sum_to_one_over_every_pair(self):
        actual = np.array([0] * 20 + [1] * 20)
        predicted = np.array([1] * 6 + [0] * 14 + [0] * 9 + [1] * 11)
        matrix = scores(actual, predicted)["confusion_matrix"]
        pairs = confusion_pairs(matrix, top=len(CLASSES) ** 2)
        self.assertAlmostEqual(sum(e["share_of_errors"] for e in pairs), 1.0, places=6)

    def test_a_perfect_prediction_has_no_pairs_with_a_count(self):
        actual = np.arange(len(CLASSES)).repeat(4)
        matrix = scores(actual, actual)["confusion_matrix"]
        self.assertEqual(sum(e["count"] for e in confusion_pairs(matrix)), 0)


class ModelDefinition(unittest.TestCase):
    def test_the_parameter_count_is_what_the_readme_quotes(self):
        from src.model import CIFAR10CNN
        self.assertEqual(sum(p.numel() for p in CIFAR10CNN().parameters()), 815_018)

    def test_it_produces_one_logit_per_class(self):
        import torch

        from src.model import CIFAR10CNN
        out = CIFAR10CNN()(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (2, len(CLASSES)))


class RecordedMetricsFollowFromRecordedPredictions(unittest.TestCase):
    """The check that needs no GPU and catches a hand-edited number."""

    @classmethod
    def setUpClass(cls):
        if not RESULTS.is_file():
            raise unittest.SkipTest("no recorded results yet")
        cls.data = json.loads(RESULTS.read_text(encoding="utf-8"))

    def test_every_stored_metric_recomputes(self):
        for name, entry in self.data["models"].items():
            if "predictions" not in entry:
                continue
            with self.subTest(model=name):
                stored = entry["predictions"]
                fresh = scores(np.array(stored["test_true"]),
                               np.array(stored["test_predicted"]))
                self.assertAlmostEqual(fresh["top1_accuracy"],
                                       entry["test"]["top1_accuracy"], places=12)
                self.assertAlmostEqual(fresh["macro_f1"],
                                       entry["test"]["macro_f1"], places=12)
                self.assertEqual(fresh["confusion_matrix"],
                                 entry["test"]["confusion_matrix"])
                for label, values in fresh["per_class"].items():
                    for key, value in values.items():
                        self.assertAlmostEqual(
                            value, entry["test"]["per_class"][label][key], places=12)

    def test_the_predictions_cover_the_whole_test_split(self):
        for name, entry in self.data["models"].items():
            if "predictions" not in entry:
                continue
            with self.subTest(model=name):
                self.assertEqual(len(entry["predictions"]["test_predicted"]),
                                 self.data["data"]["test"])
                self.assertEqual(
                    set(np.unique(entry["predictions"]["test_predicted"]))
                    - set(range(len(CLASSES))), set())

    def test_the_recorded_confusion_pairs_follow_from_the_matrix(self):
        for name, entry in self.data["models"].items():
            with self.subTest(model=name):
                self.assertEqual(
                    confusion_pairs(entry["test"]["confusion_matrix"])[0],
                    entry["confusion_pairs"][0])


if __name__ == "__main__":
    unittest.main()
