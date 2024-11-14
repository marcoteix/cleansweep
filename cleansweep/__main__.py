
if __name__ == '__main__':

    from cleansweep.cli.filter import FilterCmd
    import argparse

    parser = argparse.ArgumentParser("CleanSweep")

    # Register args
    cmd = FilterCmd()
    cmd.add_arguments(parser)

    args = parser.parse_args()
    cmd.run(**vars(args))

