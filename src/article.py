from newspaper import Article

from codebase import Codebase


CODE_DOMAINS =  [
    "github.com",
    # "bitbucket.org",
    # "gitlab.com"
]

class CodeArticle(Article):
    def __init__(self, 
                 url: str, 
                 code_domains: list[str] = CODE_DOMAINS,
                 download: bool = True,
                 parse: bool = True):
        super().__init__(url, keep_article_html=True)
        
        self.code_domains = code_domains
        
        if download:
            self.download()
        if parse:
            self.parse()

    def code_urls(self):
        results = set()
        for elem, key, url, _ in self.clean_top_node.iterlinks():
            if elem.tag != "a":
                continue
            assert key == "href"
            
            if any(domain in url for domain in self.code_domains):
                codebase = Codebase(url)
                if codebase.repository_url not in results:
                    yield codebase
                    results.add(codebase.repository_url)
    
    