from cleansweep.cli.filter import FilterCmd
from cleansweep.cli.prepare import PrepareCmd
from cleansweep.cli.inspect import InspectCmd
from cleansweep.cli.collection import CollectionCmd
from cleansweep.cli.align import AlignCommand
from cleansweep.cli.commands import add_subcommand
from cleansweep.__version__ import __version__
import argparse
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser("CleanSweep")

    parser.add_argument(
        "--version",
        action = "version",
        help = "Prints version and exits.",
        version = f"%(prog)s v{__version__}"
    )

    filter_cmd = FilterCmd()
    prepare_cmd = PrepareCmd()
    inspect_cmd = InspectCmd()
    collection_cmd = CollectionCmd()
    align_cmd = AlignCommand()

    subparsers = parser.add_subparsers()

    for name, cmd in zip(
        [
            "filter",
            "prepare",
            "inspect",
            "collection",
            "align"
        ],
        [
            filter_cmd,
            prepare_cmd,
            inspect_cmd,
            collection_cmd,
            align_cmd
        ]
    ):
        cmd_parser = add_subcommand(
            name = name,
            subcommand = cmd,
            subparsers = subparsers
        )
        cmd.add_arguments(cmd_parser)

    args = parser.parse_args()

    kwargs = deepcopy(vars(args))

    if args.command == "filter":
        filter_cmd.run(**kwargs)
    elif args.command == "prepare":
        prepare_cmd.run(**kwargs)
    elif args.command == "inspect":
        inspect_cmd.run(**kwargs)
    elif args.command == "collection":
        collection_cmd.run(**kwargs)
    elif args.command == "align":
        align_cmd.run(**kwargs)

if __name__ == "__main__":
    main()