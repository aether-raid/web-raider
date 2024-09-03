from enum import Enum
import requests
import base64
from urllib.parse import urlparse, urlunparse

class CodebaseType(str, Enum):
    GITHUB = "github"
    # GITLAB = "gitlab"
    # BITBUCKET = "bitbucket"
    
def format_type(type: CodebaseType):
    return type[0].upper()+type[1:3]+type[3].upper()+type[4:]

class Codebase:
    # def __new__(cls, url: str):
    #     parsed_url = urlparse(url)
    #     domain = parsed_url.netloc.lower()
        
    #     if "github.com" in domain:
    #         return GitHubCodebase.__new__(cls)
    #     elif "gitlab.com" in domain:
    #         return super(Codebase, cls).__new__(cls)
    #     elif "bitbucket.org" in domain:
    #         return super(Codebase, cls).__new__(cls)
    #     else:
    #         raise ValueError("Unsupported codebase type")
        
    def __init__(self, url: str):
        self.original_url = url
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if "github.com" in domain:
            self.type = CodebaseType.GITHUB
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        # elif "gitlab.com" in domain:
        #     self.type = CodebaseType.GITLAB
        #     path_parts = path.split('/')[:3]  # Keep only username and repo name
        # elif "bitbucket.org" in domain:
        #     self.type = CodebaseType.BITBUCKET
        #     path_parts = path.split('/')[:3]  # Keep only username and repo name
        else:
            raise ValueError("Unsupported codebase type")

        # Reconstruct the URL with only the repository part
        truncated_path = '/'.join(path_parts)
        truncated_parsed_url = parsed_url._replace(path=truncated_path, query='', fragment='')
        self.repository_url = urlunparse(truncated_parsed_url)
    
    def __str__(self):
        return f'{format_type(self.type)}Codebase({self.repository_url})'
    
    def __repr__(self):
        return str(self)

    @staticmethod
    def is_code(url: str) -> bool:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if "github.com" in domain: # and "/blob/" not in path:
            return True
        # elif "gitlab.com" in domain: # and "/blob/" not in path:
        #     return True
        # elif "bitbucket.org" in domain and "/src/" not in path:
        #     return True
        else:
            return False


class GitHubCodebase(Codebase):
    def __init__(self, url):
        super().__init__(url)
        if self.type != CodebaseType.GITHUB:
            raise ValueError("This is not a GitHub codebase")

    def get_id(self):
        # Extract owner and repo from the repository URL
        path_parts = urlparse(self.repository_url).path.split('/')
        owner, repo = path_parts[1], path_parts[2]
        
        return owner, repo
    
    def get_topics(self):
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitHub API URL
        api_url = f"https://api.github.com/repos/{owner}/{repo}/topics"

        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.mercy-preview+json"}
        response = requests.get(api_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract and return the topics
            return response.json()["names"]
        else:
            # Handle errors (e.g., repository not found, API rate limit exceeded)
            print(f"Error fetching topics: {response.status_code}")
            return []
        
    def get_readme(self):
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitHub API URL for the README
        api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"

        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            content = response.json()["content"]
            decoded_content = base64.b64decode(content).decode('utf-8')
            return decoded_content
        else:
            print(f"Error fetching README: {response.status_code}")
            return None

    def get_repo_desc(self):
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()
        
        # Construct the GitHub API URL for the README
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()["description"]
        else:
            print(f"Error fetching repo description: {response.status_code}")
            return None


# class GitLabCodebase(Codebase):
#     def __init__(self, url):
#         parse_result = urlparse(url)
#         self.base_url = parse_result.netloc
        
#         super().__init__(url.replace(self.base_url, ))
