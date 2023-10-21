import argparse

import nimblephysics_libs.biomechanics

from cli.train import TrainCommand
from cli.visualize import VisualizeCommand
from cli.create_splits import CreateSplitsCommand
import nimblephysics as nimble

def main():
    commands = [TrainCommand(),
                VisualizeCommand(),
                CreateSplitsCommand()]

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
    main()
