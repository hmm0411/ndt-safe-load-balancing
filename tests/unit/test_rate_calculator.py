import unittest
from src.telemetry.rate_calculator import RateCalculator

class RateCalculatorTests(unittest.TestCase):
    def test_rate(self):
        c = RateCalculator()
        self.assertEqual(c.update("x", 100, 1.0), 0.0)
        self.assertAlmostEqual(c.update("x", 160, 2.0), 60.0)

    def test_reset(self):
        c = RateCalculator()
        c.update("x", 100, 1.0)
        self.assertEqual(c.update("x", 20, 2.0), 0.0)

if __name__ == "__main__":
    unittest.main()
