from dataclasses import dataclass, field


@dataclass
class Query:
    """
    Represents a user query possibly including title, author, and bibliographic info.

    Attributes:
        query_id (int): Unique identifier of the query.
        title (str): Title of the query (optional).
        author (str): Author field, if present.
        content (str): The main query content (usually from .W).
        bibliography (str): Additional source info, if any (from .B).
        
    Example JSON input:
        {
            "query_id": 1,
            "title": "Information Retrieval",
            "author": "J. Doe",
            "content": "What is information retrieval?",
            "bibliography": "Source: Journal of IR"
        }
    """
    id: int
    title: str = ""
    author: str = ""
    content: str = ""
    bibliography: str = ""
    
    
    @staticmethod
    def from_json(data: dict) -> 'Query':
        """
        Create a Query object from a JSON-like dictionary.
        
        Args:
            data (dict): Dictionary containing query data.
            
        Returns:
            Query: An instance of the Query class.
        """
        return Query(
            id=int(data["id"]),
            title=data.get("title", ""),
            author=data.get("author", ""),
            content=data.get("content", ""),
            bibliography=data.get("biblio", "")
        )