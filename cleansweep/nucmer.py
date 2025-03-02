from copy import deepcopy
from dataclasses import dataclass
import io
from pathlib import Path 
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Iterable
from cleansweep.typing import File, Directory
from Bio import SeqIO
import subprocess
import shutil
import joblib

NUCMER_COORDS_HEADER = [
    "ref_start",
    "ref_end",
    "query_start",
    "query_end",
    "ref_aln_len",
    "query_aln_len",
    "pct_identity",
    "ref_len",
    "query_len",
    "ref",
    "query"
]

NUCMER_SNPS_HEADER = [
    "pos",
    "ref",
    "alt",
    "query_pos",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "ref_id",
    "query_id"
]

@dataclass
class NucmerAlignment:

    min_identity: float = .80
    min_length: int = 150

    def prepare(
        self,
        reference: File,
        queries: List[File],
        output: Directory,
        tmp_dir: Union[None, Directory] = None,
        keep_tmp: bool = True
    ):
        
        outdir = Path(output)
        outdir.mkdir(
            exist_ok = True
        )
        
        if tmp_dir is None: 
            tmp_dir = outdir.joinpath("tmp")

        # Align queries to the reference with nucmer
        file_map = self.align(
            reference = reference,
            queries = queries,
            outdir = outdir,
            tmp_dir = tmp_dir
        )

        # Write reference FASTA to output

        output_fasta = outdir.joinpath(
            "cleansweep.reference.fa"
        )

        with open(reference) as input_file, open(output_fasta, "w") as output_file:

            reference_fasta = SeqIO.parse(input_file, "fasta")
            
            # Write to output file
            SeqIO.write(
                reference_fasta,
                output_file,
                "fasta"
            )

        # Extract contig IDs
        with open(reference) as input_file:

            reference_fasta = SeqIO.parse(input_file, "fasta")
            
            self.chrom = [
                x.name
                for x in reference_fasta
            ]

        # For each query file, mask and write to output
        for record, files in file_map.items():

            coords = self.read_coords(files["coords"])
            with open(output_fasta, "a") as output_file:
                self.mask(coords, files["fasta"], record, output_file)

        # Get a list of unaligned regions in the reference
        self.gaps = self.get_gaps(
            [x["coords"] for x in file_map.values()]
        )

        self.gaps = pd.DataFrame(
            self.gaps,
            columns = [
                "start",
                "end"
            ]
        ).set_index("start")

        # Write gaps, snps, and contig names to output
        joblib.dump(
            {
                "snps": deepcopy(self.snps),
                "gaps": deepcopy(self.gaps),
                "chrom": deepcopy(self.chrom)
            },
            outdir.joinpath(
                "cleansweep.prepare.swp"
            ),
            compress = 3
        )

        if not keep_tmp:
            shutil.rmtree(tmp_dir)

    def align(
        self,
        reference: File,
        queries: List[File],
        outdir: Directory,
        tmp_dir: Directory
    ) -> Dict[str, Dict[str, str]]:
        
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(exist_ok=True)

        self.__coords = {}
        # Holds SNPs between references
        self.snps = []

        for n, q in enumerate(queries):

            delta_file = str(tmp_dir.joinpath(f"nucmer.{n}.delta"))
            coords_file = str(tmp_dir.joinpath(f"nucmer.{n}.coords"))

            # Align with nucmer
            self.nucmer(reference, q, delta_file)
            # Convert .delta file to .coords
            self.get_coords(delta_file, coords_file)
            # Get SNPs
            self.snps.append(
                self.get_snps(delta_file)
            )

            # Extract query ID and add to the map to coords files
            for record in SeqIO.parse(q, "fasta"):
                self.__coords[record.id] = {
                    "delta": delta_file,
                    "coords": coords_file,
                    "fasta": q
                }

        self.snps = pd.concat(self.snps) \
            .reset_index()
        
        return self.__coords

    def nucmer(self, reference: File, query: File, output: File):

        command = [
            "nucmer",
            reference,
            query,
            "-p",
            output.removesuffix(".delta")
        ]

        rc = subprocess.run(command)

        if not rc.returncode == 0:
            raise RuntimeError(f"Command \"{' '.join(command)}\" failed with return code {rc.returncode}:\n{rc.stderr}")

    def get_coords(self, delta: File, output: File):

        command = [
            "show-coords",
            "-l",
            "-q",
            "-T",
            delta
        ]

        with open(output, "w") as file:
            rc = subprocess.run(command, stdout=file)

        if not rc.returncode == 0:
            raise RuntimeError(f"Command \"{' '.join(command)}\" failed with return code {rc.returncode}:\n{rc.stderr}")

    def read_coords(self, file: File) -> pd.DataFrame:

        self.coords = pd.read_csv(
            file,
            sep = "\t",
            skiprows = 4,
            names = NUCMER_COORDS_HEADER
        )

        # Extract the name of the query sequence
        self.query = self.coords["query"]

        return self.coords

    def get_mask_limits(
        self, 
        coords: pd.DataFrame,
        invert: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Marks regions in the query which align to the reference, or vice-versa (if invert is `True`).
        """
        # Exclude regions with < min_identity
        pass_coords = coords[
           coords.pct_identity.ge( 
                self.min_identity * 100
            )
        ]

        if invert:
            subject = "ref"
        else:
            subject = "query"

        # Exclude regions with < min_length
        pass_coords = pass_coords[
            pass_coords[["ref_aln_len", "query_aln_len"]] \
                .max(axis=1).ge(self.min_length)
        ]

        # Place alignments in the same orientation
        pass_coords = pass_coords.assign(
            **{
                f"f_{subject}_start": pass_coords[
                        [f"{subject}_start", f"{subject}_end"]
                    ].min(axis=1),
                f"f_{subject}_end": pass_coords[
                        [f"{subject}_start", f"{subject}_end"]
                    ].max(axis=1)
            }
        )

        # Sort alignment starts
        starts = pass_coords[f"f_{subject}_start"].sort_values().values
        # Sort alignment ends
        ends = pass_coords[f"f_{subject}_end"].sort_values().values

        # Iterate over start and ends. Track how many alignments are open
        # (open at a start; close after an end). Mask bp where this counter
        # is > 0
        counter, start_n, end_n = 0, 0, 0
        mask_starts, mask_ends = [], []

        while start_n < len(starts) and end_n < len(ends):

            if starts[start_n] < ends[end_n]: 
                if counter == 0: mask_starts.append(starts[start_n])
                counter += 1
                start_n += 1
            else:
                counter -= 1
                if counter == 0: mask_ends.append(ends[end_n])
                end_n += 1

            if counter < 0:
                raise RuntimeError(f"Something went wrong! Counter is {counter} (< 0). \
start_n: {start_n}, end_n: {end_n}, start: {starts[start_n]}, end: {ends[end_n]}.")

        return [(a, b) for a,b in zip([0]+mask_ends, mask_starts+[-1])]
            
    def mask(
        self, 
        coords: pd.DataFrame, 
        fasta: File, 
        query_id: str, 
        output: File
    ):

        self.masks = self.get_mask_limits(coords=coords)
        
        for record in SeqIO.parse(fasta, "fasta"):

            if record.id == query_id:
                masked_record = record 

                masked_record.seq = self.mask_string(str(record.seq), self.masks)
                SeqIO.write(masked_record, output, "fasta")
                break

            raise RuntimeError(f"Found no record in the FASTA file with ID {query_id}.")

    def mask_string(self, s: str, masks: List[Tuple[int, int]]) -> str:

        masked_s = []
        for start, end in masks:
            masked_s.append(
                s[start:end] if end > -1 else s[start:]
            )
        
        return "".join(masked_s)
    
    def get_snps(
        self,
        delta: File
    ):
        
        cmd = [
            "show-snps",
            "-T",
            "-r",
            str(delta)
        ]

        rc = subprocess.run(
            cmd,
            capture_output = True,
            text = True
        )

        if rc.returncode:
            raise RuntimeError(
                f"Command \"{' '.join(cmd)}\" failed with return code {rc.returncode}:\n{rc.stderr}"
            )
        
        return pd.read_table(
            io.StringIO(
                rc.stdout
            ),
            skiprows = 4,
            names = NUCMER_SNPS_HEADER
        )

    def get_gaps(
        self,
        coords: List[File]
    ) -> List[Tuple[int, int]]:
        
        # Concatenate the coords files from all queries
        coord = pd.concat(
            [
                self.read_coords(file)
                for file in coords
            ]
        )

        return self.get_mask_limits(
            coords = coord,
            invert = True
        )