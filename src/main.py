import argparse

import nimblephysics_libs.biomechanics

from cli.train import TrainCommand
from cli.visualize import VisualizeCommand
from cli.visualize_file import VisualizeFileCommand
from cli.create_splits import CreateSplitsCommand
from cli.analyze import AnalyzeCommand
from cli.sanity_check import SanityCheckCommand
from cli.make_plots import MakePlotsCommand
from cli.review_file import ReviewFileCommand
from cli.pickle_data import PickleDataCommand
import nimblephysics as nimble
import logging

def main():
    commands = [TrainCommand(),
                VisualizeCommand(),
                VisualizeFileCommand(),
                CreateSplitsCommand(),
                AnalyzeCommand(),
                SanityCheckCommand(),
                MakePlotsCommand(),
                ReviewFileCommand(),
                PickleDataCommand()]

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='InferBiomechanics Command Line Interface')

    # Split up by command
    subparsers = parser.add_subparsers(dest="command")

    # Add a parser for each command
    for command in commands:
        command.register_subcommand(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    for command in commands:
        if command.run(args):
            return


if __name__ == '__main__':
    logpath = "log"
    # Create and configure logger
    logging.basicConfig(filename=logpath,
                        format='%(asctime)s %(message)s',
                        filemode='a')

    # Creating an object
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    # Setting the threshold of logger to INFO
    logger.setLevel(logging.INFO)
    main()