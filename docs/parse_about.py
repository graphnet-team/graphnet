"""Script to parse paper.md file to About section for website."""

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

    # Get all section headers and constituent paragraphs
    sections = list(re.finditer("^# (.*?)\n", content, re.M))
    corpus = {}
    for s1, s2 in zip(sections[:-1], sections[1:]):
        corpus[s1.group(1)] = [
            par for par in content[s1.end() : s2.start()].split("\n") if par
        ]

    # Stitch together an About section
    content = "\n\n".join(
        [
            "# About",
            corpus["Summary"][1],
            "## Impact",
        ]
        + corpus["Impact on physics"]
        + [
            "## Usage",
        ]
        + corpus["Usage"][:1]
        + ["### FIGURE ###"]
        + corpus["Usage"][2:]
        + [
            "## Acknowledgements",
        ]
        + corpus["Acknowledgements"]
    )

    # Add figure
    figure = corpus["Usage"][1]
    m = re.search(
        r"!\[(?P<caption>.*) *\\label\{.*\} *\]\((?P<path>.*)\)", figure, re.M
    )
    caption, path = m.group("caption"), m.group("path")
    content = content.replace(
        "### FIGURE ###",
        f"""
:::{{figure-md}} flowchart
:class: figclass

<img src="../../paper/{path.replace(".pdf", ".png")}" alt="flowchart" width="100%">

{caption}
:::""",
    )

    # Remove references
    pattern = "\[[@\:\w]+\]"
    references = re.findall(pattern, content)
    for reference in references:
        content = content.replace(f" {reference}", "")

    # Update figure reference
    content = content.replace(
        "\\autoref{fig:flowchart}", "[the Figure](flowchart)"
    )

    # Write parsed results to output file
    with open(output_file, "w") as f:
        f.write(content)


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="""
Parse paper.md file to About section for website.
"""
    )

    parser.add_argument("-i", "--input-file", required=True)
    parser.add_argument("-o", "--output-file", required=True)

    args = parser.parse_args()

    main(args.input_file, args.output_file)
