"""Unit tests for `graphnet.utilities.argparse` module."""

import unittest

from graphnet.constants import CONFIG_DIR
from graphnet.utilities.argparse import Options, ArgumentParser


class TestOptions(unittest.TestCase):
    """Unit tests for `Options` class."""

    def setUp(self) -> None:
        """Configure instances for test."""
        self.options = Options("option1", ("option2", 42))

    def test_contains(self) -> None:
        """Test `contains` method."""
        self.assertTrue(self.options.contains("option1"))
        self.assertTrue(self.options.contains("option2"))
        self.assertFalse(self.options.contains("option3"))

    def test_pop_default(self) -> None:
        """Test `pop_default` method."""
        self.assertEqual(self.options.pop_default("option1"), None)
        self.assertEqual(self.options.pop_default("option2"), 42)
        with self.assertRaises(AssertionError):
            self.options.pop_default("option3")

    def test_len(self) -> None:
        """Test `__len__` method."""
        self.assertEqual(len(self.options), 2)

    def test_repr(self) -> None:
        """Test `__repr__` method."""
        self.assertEqual(
            repr(self.options),
            "['option1', ('option2', 42)]",
        )


class TestArgumentParser(unittest.TestCase):
    """Unit tests for `ArgumentParser` class."""

    def setUp(self) -> None:
        """Configure instances for test."""
        self.parser = ArgumentParser()

    def test_with_standard_arguments(self) -> None:
        """Test `with_standard_arguments` method."""
        standard_arguments = (
            "dataset-config",
            "model-config",
            "gpus",
            ("max-epochs", 7),
            ("batch-size", 21),
            "num-workers",
        )
        self.parser.with_standard_arguments(*standard_arguments)

        args = self.parser.parse_args(
            ["--gpus", "2", "3", "--num-workers", "0"]
        )
        self.assertEquals(tuple(args.gpus), (2, 3))
        self.assertEquals(args.num_workers, 0)
        self.assertEquals(args.max_epochs, 7)
        self.assertEquals(args.batch_size, 21)
        self.assertEquals(
            args.dataset_config,
            f"{CONFIG_DIR}/datasets/training_example_data_sqlite.yml",
        )
        self.assertEquals(
            args.model_config,
            f"{CONFIG_DIR}/models/example_energy_reconstruction_model.yml",
        )

        with self.assertRaises(AttributeError):
            args.early_stopping_patience

        # Check that unresolved arguments result in an AssertionError
        with self.assertRaises(AssertionError):
            self.parser.with_standard_arguments(
                "nonexistent-standard-argument"
            )
