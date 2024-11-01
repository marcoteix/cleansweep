import argparse
from abc import ABCMeta, abstractmethod
import textwrap

class Subcommand(metaclass=ABCMeta):

    def __init__(self):
        pass

    def add_arguments(self, parser: argparse.ArgumentParser):
        pass 

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

def add_subcommand(name: str, subcommand: Subcommand, 
    subparsers: argparse._SubParsersAction, **kwargs):

    if hasattr(subcommand.__class__, '__doc__'):
        subcommand_doc = subcommand.__class__.__doc__
        first_help_line = subcommand_doc.strip().split('\n\n')[0].strip()

        kwargs['help'] = first_help_line
        kwargs['description'] = textwrap.dedent(subcommand_doc)

    return subparsers.add_parser(name, **kwargs)