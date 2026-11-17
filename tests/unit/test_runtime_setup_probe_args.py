import unittest

from src.experiments.topologies.runtime_4c40s import (
    build_arg_parser,
    setup_probe_args_error,
)

FULL = [
    "--setup-probe-dir", "out", "--setup-probe-run-id", "r1",
    "--setup-probe-rate", "2", "--setup-probe-timeout", "1",
    "--hold-seconds", "60",
]


class SetupProbeArgsTest(unittest.TestCase):
    def error(self, argv: list[str]) -> str | None:
        return setup_probe_args_error(build_arg_parser().parse_args(argv))

    def test_probe_off_needs_nothing(self) -> None:
        self.assertIsNone(self.error([]))

    def test_full_set_is_accepted(self) -> None:
        self.assertIsNone(self.error(FULL))

    def test_rate_and_timeout_have_no_default(self) -> None:
        message = self.error(["--setup-probe-dir", "out", "--hold-seconds", "60"])
        assert message is not None
        self.assertIn("--setup-probe-rate", message)
        self.assertIn("--setup-probe-timeout", message)
        self.assertIn("--setup-probe-run-id", message)

    def test_probe_needs_a_duration(self) -> None:
        message = self.error(FULL[:-2])
        assert message is not None
        self.assertIn("--hold-seconds", message)


if __name__ == "__main__":
    unittest.main()
