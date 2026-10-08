# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('../products/debug-shared-linux-x86_64/packages/'))
# sys.path.insert(0, os.path.abspath('../../pyre/products/debug-shared-linux-x86_64/packages/'))

# -- Project information -----------------------------------------------------

project = 'AlTar'
copyright = '2013-present ParaSim Inc., 2010-present California Institute of Technology'
author = 'AlTar Development Team'

# The full version, including alpha/beta/rc tags
release = '2.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension', # the api reference, generated from the sources
    'sphinx.ext.autodoc', # import the modules
    #'sphinx.ext.autosectionlabel', # auto label sections
    'nbsphinx', # include jupyter notebooks
    #'recommonmark', # include markdown
    'myst_parser', # markdown support
    'sphinx.ext.mathjax', # render math via JavaScript, another option is sphinx.ext.imgmath
]

# needed for readthedocs
master_doc = 'index'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build',
                    'Thumbs.db', '.DS_Store',
                    'api-gen', # ignore api reference generators
                    '**.ipynb_checkpoints', # jupyter notebook progress
                    ]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# -- the api reference --------------------------------------------------------
# sphinx-autoapi reads the sources without importing them, so the docs build without compiling
# altar; the sources are laid out as they are installed, with symlinks, before it reads them
import pathlib
import shutil


def _stage_sources():
    """
    Mirror the installed package: altar/altar as {altar}, altar/cuda as {altar.cuda}, and each
    models/<name>/<name> as {altar.models.<name>}
    """
    docs = pathlib.Path(__file__).resolve().parent
    root = docs.parent
    stage = docs / "_autoapi_src" / "altar"
    shutil.rmtree(stage.parent, ignore_errors=True)
    (stage / "models").mkdir(parents=True)
    for entry in (root / "altar" / "altar").iterdir():
        if entry.name not in ("models", "__pycache__"):
            (stage / entry.name).symlink_to(entry)
    (stage / "cuda").symlink_to(root / "altar" / "cuda")
    for entry in (root / "altar" / "altar" / "models").iterdir():
        if entry.name != "__pycache__":
            (stage / "models" / entry.name).symlink_to(entry)
    for model in sorted((root / "models").iterdir()):
        package = model / model.name
        if (package / "__init__.py").exists():
            (stage / "models" / model.name).symlink_to(package)
    return stage


autoapi_dirs = [str(_stage_sources())]
autoapi_follow_symlinks = True
autoapi_root = 'api'
autoapi_add_toctree_entry = False # {index} lists api/index itself
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary']
# the pre-2.0 cuda layer, superseded by the native/cuda implementations in each package
autoapi_ignore = [f'*/_autoapi_src/altar/cuda/{legacy}/*'
                  for legacy in ('bayesian', 'data', 'distributions', 'models', 'norms')]
# compiled extension modules have no python source to resolve imports into
suppress_warnings = ['autoapi.python_import_resolution']

# -- jupyter notebooks ------------------------------------------------------------
# rendered with their saved outputs; running them needs altar, and a gpu for some
nbsphinx_execute = 'never'

# markdown (MyST): math with $...$ and $$...$$, amsmath environments, definition lists
myst_enable_extensions = ["dollarmath", "amsmath", "deflist"]

# for markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# number the equations of amsmath environments, as LaTeX does
mathjax3_config = {
    'tex': {'tags': 'ams'},
}

# --- latex pdf ---------
latex_engine = 'xelatex'
latex_elements = {
}
#latex_show_urls = 'footnote'
