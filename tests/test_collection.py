import logging
import unittest
import subprocess
from pathlib import Path

# Path to test files
data_dir = Path("tests/data/collection")

# Output directory
outdir = Path("tests/outputs/collection")
outdir.mkdir(
    parents = True,
    exist_ok = True
)

class TestCleanSweepCollectionCLI(unittest.TestCase):

    def test_correct_input(self):

        cmd = [
            "cleansweep",
            "collection"
        ] + [
            str(x) 
            for x in data_dir.glob("test.*.vcf")
        ] + [
            "--output",
            str(
                outdir.joinpath("test_correct_input.vcf")
            ),
            "--tmp-dir",
            str(
                outdir.joinpath("tmp")
            ),
            "-a", "0.998"
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

if __name__ == '__main__':
    unittest.main()