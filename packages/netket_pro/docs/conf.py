# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Set env variables to correctly detect sphinx in NetKet
import os
import sys
import pathlib

os.environ["NETKET_SPHINX_BUILD"] = "1"
import netket as nk
import netket_pro as nkp

# add the folder with sphinx extensions
sys.path.append(str(pathlib.PosixPath(os.getcwd()) / "sphinx_extensions"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NetKet Fidelity"
copyright = "2024, Filippo Vicentini & Collaborators"
author = "Filippo Vicentini & Collaborators"
release = nkp.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.graphviz",
    "sphinx_rtd_theme",
    # "custom_inheritance_diagram.inheritance_diagram",  # this is a custom patched version because of bug sphinx#2484
    "flax_module.fmodule",
]
# Add any paths that contain templates here, relative to this directory.
templates_path = [
    "templates",
    "templates/autosummary",
    "templates/sections",
    "_templates",
]

# For sphinx.ext.linkcode
from link_to_source import linkcode_resolve


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Napoleon settings
autodoc_docstring_signature = True
autodoc_inherit_docstrings = True
allow_inherited = True
autosummary_generate = True
napoleon_preprocess_types = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
# source_suffix = {
#    ".rst": "restructuredtext",
#    ".ipynb": "myst-nb'",
#    ".md": "markdown",
#    '.myst': 'myst-nb',
# }
source_suffix = [".rst", ".ipynb", ".md"]

# Markdown parser latex support
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "html_admonition"]
myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

myst_heading_anchors = 2
autosectionlabel_maxdepth = 1

# -- Pre-process -------------------------------------------------
autodoc_mock_imports = ["openfermion", "qutip"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "netket": ("https://netket.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "qutip": ("https://qutip.readthedocs.io/en/latest/", None),
    "pyscf": ("https://pyscf.org/", None),
}


# do not show __init__ if it does not have a docstring
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Ref: https://stackoverflow.com/a/21449475/
    exclusions = (
        "__weakref__",  # special-members
        "__doc__",
        "__module__",
        "__dict__",  # undoc-members
        "__new__",
    )
    exclude = name in exclusions
    if name == "__init__":
        exclude = True if obj.__doc__ is None else False
    return True if (skip or exclude) else None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect('autodoc-process-docstring', warn_undocumented_members);

    # fix modules
    # process_module_names(netket)
    # process_module_names(netket.experimental)
