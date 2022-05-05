import argparse
from typing import List, NamedTuple, Optional, Union

def parse_args(args: Optional[Union[str, List[str]]] = None) -> NamedTuple:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Simple example to construct and train a graph neural network using "
            "GraphNeT"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional / required arguments
    parser.add_argument(
        "database",
        help="database path",
    )
    parser.add_argument(
        "output",
        help="output directory name",
    )

    # Optional arguments
    parser.add_argument(
        "--pulsemap", "-p",
        help="pulsemap to be extracted",
        default='SRTTWOfflinePulsesDC',
    )
    parser.add_argument(
        "--target", "-t",
        help="reconstruction target",
        default='energy',
    )
    parser.add_argument(
        "--batch", "-b",
        help="batch size for training",
        default=512,
        type=int,
    )
    parser.add_argument(
        "--epochs", "-e",
        help="number of epochs for training",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--workers", "-w",
        help="number of workers used for DataLoader",
        default=10,
    )
    parser.add_argument(
        "--gpu", "-g",
        help="index of GPU to use",
        choices=[1,2],
        type=int,
    )
    parser.add_argument(
        "--patience",
        help=(
            "number of epochs to wait for decrease in validation loss before "
            "terminating training early"
        ),
        default=5,
        type=int,
    )
    return parser.parse_args(args)


def main(args: NamedTuple) -> str:
    return args.database

    
if __name__ == "__main__":
    arguments = parse_args()
    print(main(arguments))