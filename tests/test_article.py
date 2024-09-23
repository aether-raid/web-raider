# tests/test_article.py

import unittest

from src.article import CodeArticle
from src.codebase import Codebase

ARTICLE_URL = "https://docs.r2devops.io/blog/writing-vscode-extension-with-react/"


class TestArticle(unittest.TestCase):
    def setUp(self):
        self.url = ARTICLE_URL
    
    def test_article_init(self):
        CodeArticle(self.url)
    
    def test_article_code_urls(self):
        article = CodeArticle(self.url)
        
        for codebase in article.code_urls():
            assert isinstance(codebase, Codebase)
        
    def test_article_iterator_loopback(self):
        results = []
        
        article = CodeArticle(self.url)
        
        for codebase in article.code_urls():
            results.append(codebase.repository_url)
        
        idx = 0
        
        for codebase in article.code_urls():
            assert idx < len(results)
            assert codebase.repository_url == results[idx]
            idx += 1
        
        assert idx == len(results)
    
    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()