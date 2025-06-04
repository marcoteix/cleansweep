import unittest
import subprocess
from pathlib import Path

# Path to test files
references_dir = Path("tests/data/references")

incomplete_background_fastas = list( 
    references_dir.glob("incomplete/background.*.incomplete.fa")
)
incomplete_reference_fasta = references_dir.joinpath(
    "incomplete",
    "query.Esch_coli_TUM2802.incomplete.fa"
)

background_fastas = list( 
    references_dir.glob("complete/background.*.fa")
)
reference_fasta = references_dir.joinpath(
    "complete",
    "query.Esch_coli_TUM2802.fa"
)

background_fastas_gzip = list( 
    references_dir.glob("complete/background.*.fa.gz")
)
reference_fasta_gzip = references_dir.joinpath(
    "complete",
    "query.Esch_coli_TUM2802.fa.gz"
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

        cmd = [
            "cleansweep",
            "prepare",
            str(incomplete_reference_fasta),
            "--background"
        ] + [ 
            str(x)
            for x in incomplete_background_fastas
        ] + [
            "-o",
            str(
                outdir.joinpath("test_incomplete_references")
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

    def test_all_fastas_gzip(self):
        
        cmd = [
            "cleansweep",
            "prepare",
            str(reference_fasta_gzip),
            "--background"
        ] + [ 
            str(x)
            for x in background_fastas_gzip
        ] + [
            "-o",
            str(
                outdir.joinpath("test_all_fastas_gzip")
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