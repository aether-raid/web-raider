from enum import Enum
import os
import requests
import base64
import re
from urllib.parse import urlparse, urlunparse
from assets.key_import import GITHUB_TOKEN

class CodebaseType(str, Enum):
    """
    A class to define a set of codebase types by extending `Enum`.

    Members
    -------
    GITHUB: str
        represents GitHub   
    GITLAB: str
        represents GitLab
    BITBUCKET: str
        represents BitBucket
    SOURCEFORGE: str
        represents SourceForge
    GITEE: str
        represents Gitee
    """
    GITHUB = "GitHub"
    GITLAB = "GitLab"
    BITBUCKET = "BitBucket"
    SOURCEFORGE = "SourceForge"
    GITEE = "Gitee"

    @classmethod
    def format_type(cls, codebase_type: 'CodebaseType') -> str:
        """
        Format the codebase type to a specific string representation.

        Parameters
        ----------
        codebase_type : CodebaseType
            The codebase type to format.

        Returns
        -------
        str
            The formatted string representation of the codebase type.
        """
        """
        Format the codebase type to a specific string representation.

        Parameters
        ----------
        codebase_type : CodebaseType
            The codebase type to format.

        Returns
        -------
        str
            The formatted string representation of the codebase type.
        """
        return codebase_type.value

class Codebase:
    """
    A class that holds the required information of codebases.

    Attributes
    ----------
    original_url: str
        The original URL of the repository.
    repository_url: str
        The URL of the repository with only the repository part.
    type: CodebaseType
        The type of the codebase (e.g., GitHub, GitLab, etc.).

    Methods
    -------
    __new__(cls, url: str)
        Determines the type of codebase based on the domain and returns an instance of the corresponding subclass.
    
    __init__(self, url: str)
        Initializes the Codebase instance with the given URL.
    
    __str__(self)
        Returns a string representation of the Codebase instance.
    
    __repr__(self)
        Returns a string representation of the Codebase instance.
    
    is_code(url: str) -> bool
        Determines if the given URL is a code repository URL.
    """
    def __new__(cls, url: str):
        """
        Determines the type of codebase based on the domain and returns an instance of the corresponding subclass.

        Parameters
        ----------
        url : str
            The URL of the repository.

        Returns
        -------
        Codebase
            An instance of the corresponding subclass of Codebase.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        if "github.com" in domain:
            return super(Codebase, GitHubCodebase).__new__(GitHubCodebase)
        elif "gitlab.com" in domain:
            return super(Codebase, GitLabCodebase).__new__(GitLabCodebase)
        elif "bitbucket.org" in domain:
            return super(Codebase, BitBucketCodebase).__new__(BitBucketCodebase)
        elif "sourceforge.net" in domain:
            return super(Codebase, SourceForgeCodebase).__new__(SourceForgeCodebase)
        elif "gitee.com" in domain:
            return super(Codebase, GiteeCodebase).__new__(GiteeCodebase)
        else:
            raise ValueError("Unsupported codebase type")
        
    def __init__(self, url: str):
        """
        Initializes the Codebase instance with the given URL.

        Parameters
        ----------
        url : str
            The URL of the repository.
        """
        self.original_url = url
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if "github.com" in domain:
            self.type = CodebaseType.GITHUB
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "gitlab.com" in domain:
            self.type = CodebaseType.GITLAB
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "bitbucket.org" in domain:
            self.type = CodebaseType.BITBUCKET
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "sourceforge.net" in domain:
            self.type = CodebaseType.SOURCEFORGE
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        elif "gitee.com" in domain:
            self.type = CodebaseType.GITEE
            path_parts = path.split('/')[:3]  # Keep only username and repo name
        else:
            raise ValueError("Unsupported codebase type")
        
        # Reconstruct the URL with only the repository part
        truncated_path = '/'.join(path_parts)
        truncated_parsed_url = parsed_url._replace(path=truncated_path, query='', fragment='')
        self.repository_url = urlunparse(truncated_parsed_url)
    
    def __str__(self):
        """
        Returns a string representation of the Codebase instance.

        Returns
        -------
        str
            A string representation of the Codebase instance.
        """
        return f'{CodebaseType.format_type(self.type)} Codebase({self.original_url})'
    
    def __repr__(self):
        """
        Returns a string representation of the Codebase instance.

        Returns
        -------
        str
            A string representation of the Codebase instance.
        """
        return str(self)

    @staticmethod
    def is_code(url: str) -> bool:
        """
        Determines if the given URL is a code repository URL.

        Parameters
        ----------
        url : str
            The URL to check.

        Returns
        -------
        bool
            True if the URL is a code repository URL, False otherwise.
        """
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        if "github.com" in domain and 'docs.github.com' not in domain:
            return True
        elif "gitlab.com" in domain and "/-/blob/" not in path:
            return True
        elif "bitbucket.org" in domain and "/src/" not in path:
            return True
        elif "sourceforge.net" in domain and "/projects/" in path:
            return True
        elif "gitee.com" in domain and "/blob/" not in path:
            return True
        else:
            return False


class GitHubCodebase(Codebase):
    def __init__(self, url: str):
        super().__init__(url)
        if self.type != CodebaseType.GITHUB:
            raise ValueError("This is not a GitHub codebase")
        
    def check_is_repo(self) -> bool:
        # Regular expression patterns for profile and repository
        profile_pattern = r'^https://github\.com/[^/]+$'
        repo_pattern = r'^https://github\.com/[^/]+/[^/]+$'
        
        if re.match(profile_pattern, self.original_url):
            return False
        elif re.match(repo_pattern, self.original_url):
            return True
        else:
            return False

    def get_id(self):
        # Extract owner and repo from the repository URL
        # path_parts = urlparse(self.repository_url).path.split('/')
        path_parts = urlparse(self.original_url).path.split('/')
        owner, repo = path_parts[1], path_parts[2]
        
        return owner, repo
    
    def get_topics(self):
        # Extract owner and repo from the repository URL
        owner, repo = self.get_id()

        # Construct the GitHub API URL
        api_url = f"https://api.github.com/repos/{owner}/{repo}/topics"

        # Make a GET request to the GitHub API
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
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
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
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
        headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.get(api_url, headers=headers)
        
        if response.status_code == 200:
            return response.json()["description"]
        else:
            print(f"Error fetching repo description: {response.status_code}")
            return None
        
    def combine_info(self):
        # if correct type
        if self.check_is_repo():
            topics = self.get_topics() if not None else []  # Default to an empty list if None
            readme = self.get_readme() if not None else ''  # Default to an empty string if None
            # desc = self.get_repo_desc() if not None else '' # Default to an empty string if None
            
            info_dict = {
                'topics': topics,
                'readme': readme,
                # 'description': desc
            }

            return info_dict
        
        else:
            return None

class GitLabCodebase(Codebase):
    def __init__(self, url):
        super().__init__(url)
        if self.type != CodebaseType.GITLAB:
            raise ValueError("This is not a GitLab codebase")

class BitBucketCodebase(Codebase):
    def __init__(self, url):
        super().__init__(url)
        if self.type != CodebaseType.BITBUCKET:
            raise ValueError("This is not a BitBucket codebase")

class SourceForgeCodebase(Codebase):
    def __init__(self, url):
        super().__init__(url)
        if self.type != CodebaseType.SOURCEFORGE:
            raise ValueError("This is not a SourceForge codebase")

class GiteeCodebase(Codebase):
    def __init__(self, url):
        super().__init__(url)
        if self.type != CodebaseType.GITEE:
            raise ValueError("This is not a Gitee codebase")
