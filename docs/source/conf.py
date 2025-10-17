import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'Gower-metric'
copyright = '2025, gower-metric developers'
author = 'gower-metric developers'

release = '1.0.0'
version = '1.0.0'

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

# enable google and numpy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autosummary_generate_overviews = True

add_module_names = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_toc_level": 4,
    "navigation_depth": 4,
    "show_nav_level": 4,
    "collapse_navigation": True,
    "navigation_with_keys": True,
}

epub_show_urls = 'footnote'
