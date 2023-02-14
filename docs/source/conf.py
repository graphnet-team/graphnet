# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import re


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "graphnet"
copyright = f"2021-{datetime.today().year}, GraphNeT team"
author = "GraphNeT team"
title = "GraphNeT"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

toc_object_entries_show_parents = "hide"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_material"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": title,
    # Set you GA account ID to enable tracking
    "google_analytics_account": "UA-XXXXX",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://graphnet-team.github.io/graphnet",
    # Set the color and the accent color
    "color_primary": "indigo",
    "color_accent": "blue",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/graphnet-team/graphnet/",
    "repo_name": title,
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": -1,
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": True,
    "heroes": {
        "index": (
            "Graph neural networks for neutrino telescope event "
            "reconstruction."
        ),
        "install": "How to install and start using GraphNeT.",
        "contribute": "How to contribute, and the practices we follow.",
        "code": "Example scripts and API reference.",
    },
    "master_doc": False,
    "nav_links": [
        {"href": "index", "internal": True, "title": "Documentation"},
    ],
}

html_show_sourcelink = False

html_sidebars = {
    "**": [
        "logo-text.html",
        "globaltoc.html",
        "localtoc.html",
        "searchbox.html",
    ]
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "../../assets/identity/favicon-white.svg"

html_favicon = "../../assets/identity/favicon.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- Options for Napoleon ----------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


# -- Options for sphinx-autodoc ------------------------------------

special_members = []
autoclass_content = "both"
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": ", ".join(special_members) if special_members else None,
    "undoc-members": True,
    "private-members": False,
    "show-inheritance": True
}
autodoc_mock_imports = [
    "pisa",
    "icecube",
    "icecube.icetray",
    "icecube.dataclasses",
    "icetray",
    "dataclasses",
]


# -- Options for sphinx-autodoc-typehints ------------------------------------

typehints_defaults = "comma"
autodoc_typehints = "description"
autodoc_typehints_format = "short"


# -- Options for MyST --------------------------------------------------------

myst_enable_extensions = ["colon_fence"]


# -- Formatting and filtering ------------------------------------------------


def remove_default_value(
    app, what, name, obj, options, signature, return_annotation
):
    if signature:
        signature = re.sub(r"=[\w'\-\.]+", "", signature)
    return (signature, return_annotation)


def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    return (
        True
        if (name.startswith("_") and name not in special_members)
        else None
    )


def setup(app):
    app.connect("autodoc-process-signature", remove_default_value, priority=1)
    app.connect("autodoc-skip-member", autodoc_skip_member_handler, priority=1)
