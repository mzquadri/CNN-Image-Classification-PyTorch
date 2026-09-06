"""Tests for the ways a CIFAR-10 accuracy can be measuring the wrong thing.

The failures worth guarding against in image classification are quiet ones. An
augmented validation set scores lower than it should and nobody notices. A
duplicate across splits scores higher and nobody notices either. Both are checked
here, and each check that something is clean is paired with one showing the same
probe fires when it is not.

Anything needing the dataset skips itself when CIFAR-10 is not present, so the
suite runs on a machine that has never downloaded it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.audit import content_hashes
from src.benchmark import CLASSES, DATA, SEED, SUBSET_SIZE

CIFAR_PRESENT = (DATA / "cifar-10-batches-py").is_dir()


def requires_cifar(test):
    return unittest.skipUnless(
        CIFAR_PRESENT, "CIFAR-10 is not downloaded")(test)


class ContentHashing(unittest.TestCase):
    """The probe used to find duplicates has to be able to find one."""

    def test_identical_arrays_hash_alike(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.assertEqual(content_hashes([image, image.copy()])[0],
                         content_hashes([image, image.copy()])[1])

    def test_a_single_changed_pixel_changes_the_hash(self):
        first = np.zeros((32, 32, 3), dtype=np.uint8)
        second = first.copy()
        second[0, 0, 0] = 1
        hashes = content_hashes([first, second])
        self.assertNotEqual(hashes[0], hashes[1])

    def test_a_planted_duplicate_is_counted(self):
        rng = np.random.default_rng(0)
        train = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(20)]
        test = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(10)]
        test[3] = train[7].copy()
        known = set(content_hashes(train))
        found = sum(1 for value in content_hashes(test) if value in known)
        self.assertEqual(found, 1)


class NearDuplicateSearch(unittest.TestCase):
    """The mirrored search is the one that found anything, so it is pinned."""

    def setUp(self):
        from src.audit import nearest_neighbour_distances
        self.search = nearest_neighbour_distances
        rng = np.random.default_rng(1)
        self.train = rng.integers(0, 255, (40, 32, 32, 3), dtype=np.uint8)

    def test_an_identical_image_is_found_at_distance_zero(self):
        test = np.stack([self.train[5], self.train[11]])
        distances, indices = self.search(test, self.train, include_flips=False)
        np.testing.assert_allclose(distances, [0.0, 0.0], atol=1e-9)
        self.assertEqual(list(indices), [5, 11])

    def test_a_mirrored_image_is_missed_without_flips_and_found_with_them(self):
        """The whole reason the audit runs the search twice."""
        mirrored = np.stack([self.train[9][:, ::-1, :]])
        without, _ = self.search(mirrored, self.train, include_flips=False)
        with_flips, indices = self.search(mirrored, self.train, include_flips=True)
        self.assertGreater(without[0], 5.0)
        self.assertLess(with_flips[0], 1e-9)
        self.assertEqual(indices[0], 9)

    def test_unrelated_images_are_far_apart(self):
        rng = np.random.default_rng(2)
        test = rng.integers(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        distances, _ = self.search(test, self.train, include_flips=True)
        self.assertGreater(distances.min(), 5.0)


class Splits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CIFAR_PRESENT:
            raise unittest.SkipTest("CIFAR-10 is not downloaded")
        from src.benchmark import splits
        cls.train, cls.validation, cls.test, cls.train_indices = splits()

    @requires_cifar
    def test_the_splits_are_the_documented_sizes(self):
        self.assertEqual(len(self.train), 13_500)
        self.assertEqual(len(self.validation), 1_500)
        self.assertEqual(len(self.test), 10_000)

    @requires_cifar
    def test_train_and_validation_do_not_share_an_image(self):
        overlap = set(self.train_indices) & set(self.validation.indices)
        self.assertEqual(overlap, set())

    @requires_cifar
    def test_together_they_are_the_seeded_subset(self):
        self.assertEqual(len(set(self.train_indices) | set(self.validation.indices)),
                         SUBSET_SIZE)

    @requires_cifar
    def test_validation_is_not_augmented(self):
        """The failure this guards against is silent: an augmented validation set
        simply scores lower, and nothing says why.

        Reading the same item twice must give identical tensors. A random crop or
        a colour jitter in the pipeline would make them differ.
        """
        import torch
        first, _ = self.validation[0]
        second, _ = self.validation[0]
        self.assertTrue(torch.equal(first, second))

    @requires_cifar
    def test_training_is_augmented(self):
        """Otherwise the test above would pass on a pipeline with no augmentation
        anywhere, which is a different bug rather than the absence of one."""
        import torch
        differed = False
        for index in range(8):
            first, _ = self.train[index]
            second, _ = self.train[index]
            if not torch.equal(first, second):
                differed = True
                break
        self.assertTrue(differed, "no augmentation is being applied to training")

    @requires_cifar
    def test_the_test_split_is_not_augmented_either(self):
        import torch
        first, _ = self.test[0]
        second, _ = self.test[0]
        self.assertTrue(torch.equal(first, second))

    @requires_cifar
    def test_the_split_is_reproducible_from_the_seed(self):
        from src.benchmark import splits
        _, _, _, again = splits()
        self.assertEqual(self.train_indices, again)


class ClassBalance(unittest.TestCase):
    @requires_cifar
    def test_the_published_test_split_is_balanced(self):
        import collections

        from torchvision import datasets
        test = datasets.CIFAR10(root=str(DATA), train=False, download=False)
        counts = collections.Counter(test.targets)
        self.assertEqual(set(counts.values()), {1_000})
        self.assertEqual(len(counts), len(CLASSES))

    @requires_cifar
    def test_the_seeded_subset_is_close_to_balanced(self):
        """Not exactly balanced. It is a random draw, and that is worth pinning."""
        import collections

        import torch
        from torchvision import datasets
        train = datasets.CIFAR10(root=str(DATA), train=True, download=False)
        chosen = torch.randperm(
            len(train.data), generator=torch.Generator().manual_seed(SEED)
        )[:SUBSET_SIZE].tolist()
        counts = collections.Counter(np.array(train.targets)[chosen])
        spread = max(counts.values()) - min(counts.values())
        self.assertLess(spread / (SUBSET_SIZE / len(CLASSES)), 0.15)
        self.assertGreater(spread, 0)


class RecordedResults(unittest.TestCase):
    """The committed results file has to describe what the code produces."""

    @classmethod
    def setUpClass(cls):
        import json
        path = Path(__file__).resolve().parents[1] / "results" / "benchmark.json"
        if not path.is_file():
            raise unittest.SkipTest("no recorded results yet")
        cls.data = json.loads(path.read_text(encoding="utf-8"))

    def test_the_recorded_splits_match_the_documented_sizes(self):
        self.assertEqual(self.data["data"]["train"], 13_500)
        self.assertEqual(self.data["data"]["validation"], 1_500)
        self.assertEqual(self.data["data"]["test"], 10_000)

    def test_the_audit_found_no_exact_cross_split_duplicates(self):
        exact = self.data["audit"]["exact_duplicates"]
        self.assertEqual(exact["test_images_identical_to_a_train_image"], 0)

    def test_the_mirrored_search_found_more_than_the_plain_one(self):
        """If it ever stops doing so, the README's reason for running both is gone."""
        near = self.data["audit"]["near_duplicates"]
        self.assertLess(near["pixels_or_mirror_image"]["closest_pair_distance"],
                        near["pixels"]["closest_pair_distance"])

    def test_the_reference_run_reproduces_within_a_point(self):
        reference = self.data["reference_run"]
        self.assertLess(abs(reference["difference_points"]), 1.0)
        self.assertTrue(reference["parameters_match"])


if __name__ == "__main__":
    unittest.main()
