#%%

import unittest
import subprocess
from pathlib import Path

# Path to test files
data_dir = Path("tests/data/align")

# Output directory
outdir = Path("tests/outputs/align")
outdir.mkdir(
    parents = True,
    exist_ok = True
)

class TestCleanSweepCollectionCLI(unittest.TestCase):

    def test_correct_inputs(self):

        cmd = [
            "cleansweep",
            "align"
        ] + [
            str(data_dir.joinpath("test.1.tiny.fastq.gz")),
            str(data_dir.joinpath("test.2.tiny.fastq.gz")) 
        ] + [
            "--reference", str(data_dir.joinpath("cleansweep.reference.fa")),
            "--output", str(outdir.joinpath("test_correct_inputs.bam"))
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

if __name__ == '__main__':
    unittest.main()
# %%
