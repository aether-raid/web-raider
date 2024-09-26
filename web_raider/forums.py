# src/forums.py

from stackapi import StackAPI
from enum import Enum
from urllib.parse import urlparse

class ForumType(str, Enum):
    STACKOVERFLOW = "StackOverflow"

class Forum:
    def __new__(cls, url: str) -> 'Forum':
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        if 'stackoverflow.com' in domain:
            return super(Forum, StackOverflowForum).__new__(StackOverflowForum)

    def __init__(self, url):
        self.url = url

class StackOverflowForum(Forum):
    def __init__(self, url):
        super().__init__(url)
        self.type = ForumType.STACKOVERFLOW
        self.question_id = urlparse(url).path.lower().split('/')[2]

        self.SITE = StackAPI('stackoverflow')

    def get_answer_ids(self):
        answers = self.SITE.fetch('questions/{ids}/answers', ids=[int(self.question_id)], sort='votes')['items']
        
        # sort criteria:
        # 1. push the accepted answer to the front
        # 2. sort the rest of the answers by their upvotes (ie. score)
        answers.sort(key = lambda dictionary: (dictionary['is_accepted'] == True), reverse=True)
        
        self.answer_ids = []
        for answer in answers:
            self.answer_ids.append(answer['answer_id'])

    def get_answer_body(self, answer_id):
        # filter here to specify that we want the object to be returned along with the body
        # https://stackapps.com/a/3761
        answer = self.SITE.fetch('answers/{ids}', ids=[int(answer_id)], filter='withbody')['items']

        # need to parse the body AHHHHHH
        return answer[0]['body']
    
