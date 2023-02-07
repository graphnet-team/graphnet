"""Script to parse README.md file to installation instructions for website."""

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

    # Remove anything before "Install" section
    pattern = r"^##.* Install.*$\n"
    m = re.search(pattern, content, re.M)
    content = "\n".join(["# Install", content[m.end() :]])

    # Remove everying after "Install"
    pattern = r"##.*$"
    m = next(re.finditer(pattern, content, re.M))
    content = content[: m.start()]

    # Convert relevant <details>-blocks to headers
    pattern = (
        r"<details>\n" r"<summary><b>(.*?)<\/b><\/summary>\n" r"<blockquote>"
    )
    for m in re.finditer(pattern, content, re.M):
        content = content.replace(m.group(0), "## " + m.group(1))

    content = content.replace("</blockquote>\n</details>", "")

    # Trim for whitespaces and newlines
    content = content.strip()
    for _ in range(2):
        content = content.replace("\n\n\n", "\n\n")

    # Write parsed results to output file
    with open(output_file, "w") as f:
        f.write(content)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="""
Parse README.md file to installation instructions for website.
"""
    )

    parser.add_argument("-i", "--input-file", required=True)
    parser.add_argument("-o", "--output-file", required=True)

    args = parser.parse_args()

    main(args.input_file, args.output_file)
