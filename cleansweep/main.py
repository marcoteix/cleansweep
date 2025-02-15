from cleansweep.cli.filter import FilterCmd
from cleansweep.cli.prepare import PrepareCmd
import argparse
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser("CleanSweep")

    subparsers = parser.add_subparsers()

    filter_parser = subparsers.add_parser("filter", help="Filter a VCF file.")
    filter_parser.set_defaults(command="filter")
    prepare_parser = subparsers.add_parser("prepare", help="Prepare a reference for alignment.")
    prepare_parser.set_defaults(command="prepare")

    # Register args
    filter_cmd = FilterCmd()
    filter_cmd.add_arguments(filter_parser)
    prepare_cmd = PrepareCmd()
    prepare_cmd.add_arguments(prepare_parser)

    args = parser.parse_args()

    kwargs = deepcopy(vars(args))
    if "command" in kwargs:
        del kwargs["command"]

    if args.command == "filter":
        filter_cmd.run(**kwargs)
    elif args.command == "prepare":
        prepare_cmd.run(**kwargs)

if __name__ == "__main__":
    main()