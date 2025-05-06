import unittest
import subprocess
from pathlib import Path

# Path to test files
references_dir = Path("tests/data/references")
background_fastas = list( 
    references_dir.glob("complete/background.*.fa")
)
reference_fasta = references_dir.joinpath(
    "complete",
    "query.Esch_coli_TUM2802.fa"
)

# Output directory
outdir = Path(
    "tests/outputs/prepare"
)

class TestCleanSweepPrepareCLI(unittest.TestCase):

    def test_no_background_strain(self):

        cmd = [
            "cleansweep",
            "prepare",
            str(reference_fasta),
            "--background",
            "-o",
            str(
                outdir.joinpath(
                    "test_no_background_strain"
                )
            ),
            "-V", "4",
            "-mi", "0.95"
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_no_background_option(self):

        cmd = [
            "cleansweep",
            "prepare",
            str(reference_fasta),
            "-o",
            str(
                outdir.joinpath(
                    "test_no_background_option"
                )
            ),
            "-V", "4",
            "-mi", "0.95"
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_all_fastas(self):
        
        cmd = [
            "cleansweep",
            "prepare",
            str(reference_fasta),
            "--background"
        ] + [ 
            str(x)
            for x in background_fastas
        ] + [
            "-o",
            str(
                outdir.joinpath(
                    "test_all_fastas"
                )
            ),
            "-V", "4",
            "-mi", "0.95",
            "-k"
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_incomplete_references(self):

        background_fastas_fragmented = list( 
            references_dir.glob("fragmented/background.*.fa")
        )
        reference_fasta_fragmented = references_dir.joinpath(
            "fragmented",
            "query.Esch_coli_TUM2802.fragmented.fa"
        )
        
        cmd = [
            "cleansweep",
            "prepare",
            str(reference_fasta_fragmented),
            "--background"
        ] + [ 
            str(x)
            for x in background_fastas_fragmented
        ] + [
            "-o",
            str(
                outdir.joinpath(
                    "test_incomplete_references"
                )
            ),
            "-V", "4",
            "-mi", "0.95",
            "-k"
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )


if __name__ == '__main__':
    unittest.main()