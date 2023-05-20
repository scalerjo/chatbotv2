from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter


class BaseTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "\t", "   ", "  " " ", ""]
        super().__init__(separators=separators, **kwargs)

# NOT TESTED
class GolangCodeTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Golang syntax."""

    def __init__(self, **kwargs: Any):
        """Initialize a GolangCodeTextSplitter."""
        separators = [
            "\npackage ",
            "\nfunc ",
            "\nstruct ",
            "\ngo ",
            "\nimport"
            "\n\n",
            "\n",
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)
