import unittest
from src.orchestrator.ownership_manager import OwnershipManager

class OwnershipTests(unittest.TestCase):
    def test_commit(self):
        m = OwnershipManager({"s1": "c1"}, ["c1", "c2"])
        before = m.version
        m.commit_migration("s1", "c2", 10)
        self.assertEqual(m.get_owner("s1"), "c2")
        self.assertEqual(m.version, before + 1)

if __name__ == "__main__":
    unittest.main()
