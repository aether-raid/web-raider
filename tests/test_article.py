# tests/test_article.py

from article import CodeArticle
from codebase import Codebase

ARTICLE_URL = "https://docs.r2devops.io/blog/writing-vscode-extension-with-react/"


def test_article_init():
    CodeArticle(ARTICLE_URL)

def test_article_code_urls():
    article = CodeArticle(ARTICLE_URL)
    
    for codebase in article.code_urls():
        assert isinstance(codebase, Codebase)
