import unittest
import subprocess
from pathlib import Path

# Path to test files
cleansweep_prepare_dir = Path("tests/outputs/prepare")
pilon_dir = Path("tests/data/pilon")

# Output directory
outdir = Path(
    "tests/outputs/filter"
)

# Common CleanSweep filter options
opts = [
    "-dp", "0",
    "-a", "10",
    "-r", "0",
    "-d", "50",
    "-v", "0.1",
    "-ob", "1",
    "-Nc", "1000",
    "-s", "23",
    "-nc", "1",
    "-nd", "10",
    "-nb", "0",
    "-t", "1",
    "-V", "4" 
]

class TestCleanSweepPrepareCLI(unittest.TestCase):

    def test_uniform_edge_bc(self):

        cmd = [
            "cleansweep",
            "filter",
            str(
                pilon_dir.joinpath(
                    "uniform_edge_bc.vcf.gz"
                )
            ),
            "tests/data/prepare/uniform_edge_bc.prepare.swp"
            ,
            str(
                outdir.joinpath(
                    "test_uniform_edge_bc"
                )
            )
        ] + [
            x
            if x != "50"
            else "150" 
            for x in opts
        ]

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_no_shared_regions(self):

        cmd = [
            "cleansweep",
            "filter",
            str(
                pilon_dir.joinpath(
                    "no_shared_regions.vcf.gz"
                )
            ),
            "tests/data/prepare/no_shared_regions.prepare.swp",
            str(
                outdir.joinpath(
                    "test_no_shared_regions"
                )
            )
        ] + opts
        
        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_no_background_strain(self):

        cmd = [
            "cleansweep",
            "filter",
            str(
                pilon_dir.joinpath(
                    "test.vcf.gz"
                )
            ),
            str(
                cleansweep_prepare_dir.joinpath(
                    "test_no_background_strain",
                    "cleansweep.prepare.swp"
                )
            ),
            str(
                outdir.joinpath(
                    "test_no_background_strain"
                )
            )
        ] + opts

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_complete_references(self):

        cmd = [
            "cleansweep",
            "filter",
            str(
                pilon_dir.joinpath(
                    "test.vcf.gz"
                )
            ),
            str(
                cleansweep_prepare_dir.joinpath(
                    "test_all_fastas",
                    "cleansweep.prepare.swp"
                )
            ),
            str(
                outdir.joinpath(
                    "test_complete_references"
                )
            )
        ] + opts

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_variants_option(self):

        cmd = [
            "cleansweep",
            "filter",
            str(
                pilon_dir.joinpath(
                    "test.vcf.gz"
                )
            ),
            str(
                cleansweep_prepare_dir.joinpath(
                    "test_all_fastas",
                    "cleansweep.prepare.swp"
                )
            ),
            str(
                outdir.joinpath(
                    "test_variants_option"
                )
            ),
            "--variants"
        ] + opts

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

    def test_incomplete_reference(self):

        cmd = [
            "cleansweep",
            "filter",
            str(
                pilon_dir.joinpath("test.vcf.gz")
            ),
            str(
                cleansweep_prepare_dir.joinpath(
                    "test_incomplete_references",
                    "cleansweep.prepare.swp"
                )
            ),
            str(
                outdir.joinpath(
                    "test_incomplete_references"
                )
            ),
            "--variants"
        ] + opts

        rc = subprocess.run(cmd)
        
        # Check that the command returned 0
        self.assertEqual(
            rc.returncode,
            0
        )

if __name__ == '__main__':
    unittest.main()