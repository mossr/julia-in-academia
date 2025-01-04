# -*- coding: utf-8 -*-
"""
    pygments.styles.algfordmdark
    ~~~~~~~~~~~~~~~~~~~~~~~

    :license: MIT
"""

from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Error, Text, \
     Number, Operator, Generic, Whitespace, Punctuation, Other, Literal

class AlgForDMStyle(Style):
    """
    This style is for the Algorithms for Decision Making book.
    """

    background_color = "#f8f8f8"
    default_style = ""

    styles = {
        Whitespace:                "#bbbbbb",

        Comment:                   "italic #4b6d80",
        Comment.Multiline:         "italic #4b6d80",

        Keyword:                   "bold #4b6d80",
        Keyword.Constant:          "bold #FF00FF",

        Literal:                   "bold #4b6d80",
        Number:                    "#4b6d80", # 1, 2, etc.  (73C6F2)

        Punctuation:               "#808080", # DARK
        Operator:                  "#999999",
        Operator.Word:             "bold #AA22FF",

        Name.Builtin:              "#087EB6", # Categorical(, etc.
        Name.Function:             "#087EB6", # normalize(, etc.
        Name.Namespace:            "bold #FF00FF",
        Name.Variable:             "#00598A", # Most things are variables
        Name.Decorator:            "#4b6d80", # macros

        String:                    "#8770FE",
        String.Doc:                "italic",
        String.Interpol:           "bold #BB6688",
        String.Escape:             "bold #BB6622",
        String.Regex:              "#8770FE",
        String.Symbol:             "#F3453F",
        String.Other:              "#0072B2",

        Generic.Heading:           "bold #000080",
        Generic.Subheading:        "bold #800080",
        Generic.Deleted:           "#A00000",
        Generic.Inserted:          "#00A000",
        Generic.Error:             "#FF0000",
        Generic.Emph:              "italic",
        Generic.Strong:            "bold",
        Generic.Prompt:            "bold #0F6FA3",
        Generic.Output:            "#888",
        Generic.Traceback:         "#04D",

        Error:                     "border:#FF0000",
        Other:                     "bold #FF00FF"
    }
