import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
#
#  Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jax-sbgeom'
copyright = '2026, Timo Bogaarts'
author = 'Timo Bogaarts'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The terracotta palette below comes from the thesis preamble: dark red #b33528,
# terracotta #c46248, burnt orange #c8823c, ocher #c9a32f. furo is CSS-variable
# driven, so the colours live here and css/terracotta.css only adds the few
# things variables cannot express. Light mode is the default for first-time
# visitors; see _templates/base.html.
html_theme = 'furo'
html_static_path = ['_static']

# Shared palette entries consumed by custom.css, per light/dark scheme.
# Note: furo prepends "--" to these keys itself, so they are written without it.
_TC_LIGHT = {
    "tc-border-strong": "#d9cdc7",
    "tc-box-bg": "#fbf8f7",
}
_TC_DARK = {
    "tc-border-strong": "#463b36",
    "tc-box-bg": "#241f1d",
}

html_theme_options = {
    "light_css_variables": {
        # links and headings: terracotta darkened to >=4.5:1 on white
        "color-brand-primary": "#8f3e2e",
        "color-brand-content": "#a9482f",
        "color-brand-visited": "#8a4a3c",
        # warm neutrals in place of furo's blue-grey scale
        "color-foreground-primary": "#3d3330",
        "color-foreground-secondary": "#5b504c",
        "color-foreground-muted": "#7d716c",
        "color-foreground-border": "#ddd2cc",
        "color-background-secondary": "#f7f2ef",
        "color-background-hover": "#f1e9e5",
        "color-background-hover--transparent": "#f1e9e500",
        "color-background-border": "#e6ddd8",
        "color-inline-code-background": "#f6f1ef",
        "color-highlight-on-target": "#fbf1ec",
        "color-table-border": "#e6ddd8",
        "color-table-header-background": "#f7f2ef",
        "color-card-border": "#e6ddd8",
        # sidebar
        "color-sidebar-brand-text": "#8f3e2e",
        "color-sidebar-caption-text": "#8f6b5c",
        "color-sidebar-search-border": "#d9cdc7",
        "color-sidebar-background-border": "#e6ddd8",
        # API signatures: name in terracotta, module prefix in dark ocher
        "color-api-name": "#8f3e2e",
        "color-api-pre-name": "#8f6b22",
        "color-api-keyword": "#7d716c",
        "color-problematic": "#b33528",
        **_TC_LIGHT,
    },
    "dark_css_variables": {
        # lightened terracotta, for >=7:1 on the warm dark background
        "color-brand-primary": "#e8a887",
        "color-brand-content": "#e59171",
        "color-brand-visited": "#cba38f",
        "color-foreground-primary": "#e7ded9",
        "color-foreground-secondary": "#c3b6b0",
        "color-foreground-muted": "#9c8d87",
        "color-foreground-border": "#463b36",
        "color-background-primary": "#1c1917",
        "color-background-secondary": "#241f1d",
        "color-background-hover": "#2e2725",
        "color-background-hover--transparent": "#2e272500",
        "color-background-border": "#3c332f",
        "color-inline-code-background": "#262120",
        "color-highlight-on-target": "#3a2c22",
        "color-table-border": "#3c332f",
        "color-table-header-background": "#262120",
        "color-card-border": "#3c332f",
        "color-sidebar-brand-text": "#e8a887",
        "color-sidebar-caption-text": "#c08a6f",
        "color-sidebar-search-border": "#463b36",
        "color-sidebar-background-border": "#3c332f",
        "color-api-name": "#e59171",
        "color-api-pre-name": "#d9b463",
        "color-api-keyword": "#9c8d87",
        "color-problematic": "#e07b6c",
        **_TC_DARK,
    },
}
# Admonition hues are scheme-independent: terracotta for informational,
# burnt orange for warnings, dark red for danger, sage for tips.
_ADMONITIONS = {
    "note": "#c46248",
    "seealso": "#c46248",
    "warning": "#c8823c",
    "attention": "#c8823c",
    "caution": "#c8823c",
    "danger": "#b33528",
    "error": "#b33528",
    "tip": "#4e7d6b",
    "hint": "#4e7d6b",
    "important": "#4e7d6b",
    "admonition-todo": "#c9a32f",
}
for _kind, _hex in _ADMONITIONS.items():
    _rgba = "rgba({}, {}, {}, 0.2)".format(*(int(_hex[i:i + 2], 16) for i in (1, 3, 5)))
    for _scheme in ("light_css_variables", "dark_css_variables"):
        html_theme_options[_scheme][f"color-admonition-title--{_kind}"] = _hex
        html_theme_options[_scheme][f"color-admonition-title-background--{_kind}"] = _rgba
        # generic .. admonition:: blocks
        html_theme_options[_scheme]["color-admonition-title"] = "#c46248"
        html_theme_options[_scheme]["color-admonition-title-background"] = "rgba(196, 98, 72, 0.2)"

# terracotta.css carries the theme, custom.css styles content.
html_css_files = ["css/terracotta.css", "css/custom.css"]

autodoc_default_options = {
    "members": True, # documents all members, including dataclasses fields
    "member-order": "bysource", # order members as they appear in the source code
}

typehints_use_signature = True # show typehints in the function signature.
add_module_names = False # removes the long jax_sbgeom.module.name from the docs.

nb_execution_mode = "cache"
nb_execution_cache_path = "docs/source/notebooks/.jupyter_cache"
myst_enable_extensions = [
    "dollarmath"
]
myst_dmath_double_inline = True
# require.js is injected into <head> via _templates/base.html's extrahead
# block instead of listed here -- furo renders html_js_files at the end of
# <body>, too late for the plotly notebook outputs' requirejs/AMD embedding.
html_js_files = ["js/fit-math.js"]