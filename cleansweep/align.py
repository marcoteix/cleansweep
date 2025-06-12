from dataclasses import dataclass

import numpy as np
from cleansweep.typing import File 
from typing import Union, Iterable
import subprocess
from pathlib import Path
import logging

@dataclass
class BwaMem:

    threads: int = 1
    strictness: int = 6

    def __post_init__(self):

        if not isinstance(self.threads, int):
            raise ValueError(f"Threads should be an integer, got {type(self.threads)} ({self.threads}).")
        
        if not isinstance(self.strictness, int):
            raise ValueError(f"Strictness should be an integer, got {type(self.strictness)} ({self.strictness}).")
        
        if self.threads < 1:
            raise ValueError(f"Got a non-positive number of threads ({self.threads}).")
        
        if self.strictness < 1:
            raise ValueError(f"Strictness must be positive. Got {self.strictness}.")

    def align(
        self,
        reads: Iterable[File],
        reference: File,
        output: File
    ):
        
        self.__check_file_exists(reads)
        self.__check_file_exists(reference)

        # Check that the output directory exists
        self.__check_file_exists(
            Path(output).parent
        )

        # Index fasta
        bwa_index_command = [
            "bwa", "index",
            str(reference)
        ]

        bwa_index_rc = subprocess.run(bwa_index_command)

        if bwa_index_rc.returncode != 0:
            msg = f"Indexing {str(output)} with BWA failed. Got return code {bwa_index_rc.returncode}. Command: \'{' '.join(bwa_index_command)}\'."
            logging.error(msg)
            logging.error("stout dump:")
            logging.error(bwa_index_rc.stdout)
            raise RuntimeError(msg)

        bwa_command = ["bwa", "mem"] + [
            "-B", str(int(4 * self.strictness)),
            "-O", str(int(6 * self.strictness)),
            "-E", str(int(self.strictness)),
            "-L", str(int(5 * self.strictness)),
            "-U", str(int(9 * self.strictness)),
            "-t", str(self.threads),
            "-a"
        ] + [
            str(reference)
        ] + [
            str(x) for x in reads
        ]

        print(" ".join(bwa_command))

        bwa = subprocess.Popen(
            bwa_command,
            stdout = subprocess.PIPE,
            text = True
        )

        if bwa.returncode:
            raise RuntimeError(
                f"Calling BWA MEM wih command \"{' '.join(bwa_command)}\" failed. Return code: {bwa.returncode}."
            )
        
        # Sort output
        sort_command = ["samtools", "sort"]

        sort = subprocess.Popen(
            sort_command,
            stdin = bwa.stdout,
            stdout = subprocess.PIPE
        )

        if sort.returncode:
            raise RuntimeError(
                f"Calling samtools sort wih command \"{' '.join(sort_command)}\" failed. Return code: {sort.returncode}."
            )
        
        # Compress output
        view_command = [
            "samtools",
            "view",
            "-F", "4",
            "-o", str(output)
        ]

        view = subprocess.Popen(
            view_command,
            stdin = sort.stdout,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        )

        _, errors = view.communicate()
        
        if errors:
            raise RuntimeError(
                f"Calling samtools view wih command \"{' '.join(view_command)}\" failed. Error: {errors}."
            )
        
        # Index BAM
        index_command = [
            "samtools",
            "index",
            str(output)
        ]

        index_rc = subprocess.run(index_command)

        if index_rc.returncode != 0:
            msg = f"Indexing {str(output)} with samtools failed. Got return code {index_rc.returncode}. Command: \'{' '.join(index_command)}\'."
            logging.error(msg)
            logging.error("stout dump:")
            logging.error(index_rc.stdout)
            raise RuntimeError(msg)
        
    def get_error_rate(self) -> float:

        return .75 * np.exp(-1 * np.log(4) * int(4 * self.strictness))

    def __check_file_exists(
        self,
        file: Union[Iterable[File], File]
    ):
        
        if isinstance(file, (str, Path)):
            file = [file]

        for f in file:
            if not Path(f).exists():

                raise FileNotFoundError(
                    f"File {str(f)} does not exist."
                )