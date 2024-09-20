# tests/test_web_raider.py

from ..pipeline import main

SAMPLE_QUERY = 'Can you code a VSCode Extension using React?'

QUERY = 'How do I parse Javascript AST in Python with Tree-Sitter?'

# new query for "codebase research problem" according to aloysius
CODEBASE_QUERY = """
i specifically want open source alternatives to this:

Flowith is an innovative, canvas-based AI tool designed for content generation and deep work. It allows users to interactively create and organize various types of content, including long texts, code, and images, using a visually intuitive interface.
"""

main(QUERY)