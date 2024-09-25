# src/forums.py

from enum import Enum
from urllib.parse import urlparse

class ForumType(str, Enum):
    STACKOVERFLOW = "StackOverflow"

class Forum:
    def __new__(cls, url: str) -> 'Forum':
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower

        if 'stackoverflow.com' in domain:
            return super(Forum, StackOverflowForum).__new__(StackOverflowForum)

    def __init__(self, url):
        self.url = url

class StackOverflowForum(Forum):
    pass