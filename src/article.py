# src/article.py

from newspaper import Article
from typing import Generator

from codebase import Codebase


# CODE_DOMAINS =  [
#     "github.com",
#     # "bitbucket.org",
#     # "gitlab.com"
# ]

class CodeArticle(Article):
    def __init__(self, 
                 url: str, 
                #  code_domains: list[str] = CODE_DOMAINS,
                 download: bool = True,
                 parse: bool = True):
        super().__init__(url, keep_article_html=True)
        
        # self.code_domains = code_domains
        
        if download:
            self.download()
        if parse:
            self.parse()

    def code_urls(self) -> Generator[Codebase, None, None]:
        results = set()
        for elem, key, url, _ in self.clean_top_node.iterlinks():
            if elem.tag != "a":
                continue
            assert key == "href"
            
            # if any(domain in url for domain in self.code_domains):
            if Codebase.is_code(url):
                codebase = Codebase(url)
                if codebase.repository_url not in results:
                    yield codebase
                    results.add(codebase.repository_url)
    
    