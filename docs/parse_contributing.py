"""Script to parse CONTRIBUTING.md file to contrib instructions for website."""

import argparse
import re


def main(input_file: str, output_file: str) -> None:
    """Parsing script."""
    # Check(s)
    assert input_file != output_file
    assert input_file.endswith(".md")
    assert output_file.endswith(".md")

    # Read input file contents
    with open(input_file, "r") as f:
        content = f.read()

    # Remove "Version control"-section everything after it.
    pattern = "## Version control"
    m = re.search(pattern, content)
    content = content[: m.start()]

    # Trim for whitespaces and newlines
    content = content.strip()

    # Rename title
    content = "\n".join(["# Contribute"] + content.split("\n")[1:])

    # Update relative links for absolute ones
    content = content.replace(
        "./", "../../"
    )

    # Write parsed results to output file
    with open(output_file, "w") as f:
        f.write(content)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="""
Parse CONTRIBUTING.md file to contrib instructions for website.
"""
    )

    parser.add_argument("-i", "--input-file", required=True)
    parser.add_argument("-o", "--output-file", required=True)

    args = parser.parse_args()

    main(args.input_file, args.output_file)
