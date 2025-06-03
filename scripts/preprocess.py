import os
import re
import json
import string
import math
import pickle
from collections import defaultdict
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('stopwords')

# Enhanced NLTK resource checking
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded"""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),  # Added for newer NLTK versions
        ('corpora/stopwords', 'stopwords'),
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
        except LookupError:
            print(f"NLTK '{data_name}' not found, downloading...")
            try:
                nltk.download(data_name, quiet=True)
                print(f"NLTK '{data_name}' downloaded successfully.")
            except Exception as e:
                print(f"Failed to download NLTK '{data_name}': {e}")

# Call this at module import
ensure_nltk_data()

# # Config
# ## Editable
# *directory names*
BASE_DIR_NAME = "data"
RAW_DIR_NAME = "raw"
PREPROCESSED_DIR_NAME = "preprocessed"

# *file names*
CISI_ALL_FILENAME = "cisi.all"
QRELS_FILENAME = "qrels.text"
QUERY_FILENAME = "query.text"

# ## Non-editable
DATASET_DIR = os.path.join(BASE_DIR_NAME, RAW_DIR_NAME)
PREPROCESSED_DIR = os.path.join(BASE_DIR_NAME, PREPROCESSED_DIR_NAME)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# # QRELS
def parse_qrels(file_path=os.path.join(DATASET_DIR, QRELS_FILENAME)):
    """
    Parse a qrels file and return a mapping from query IDs to lists of relevant document IDs.

    Args:
        file_path (str, optional): Path to the qrels file. Defaults to DATASET_DIR/QRELS_FILENAME.

    Returns:
        defaultdict[list]: Dictionary mapping each query ID (int) to a list of relevant document IDs (int).

    Example:
        {
            1: [101, 102, 103],
            2: [201, 202],
            ...
        }
    """
    qrels = defaultdict(list)
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                # Only the first two columns are used; others are ignored
                qrels[query_id].append(doc_id)
    return qrels

# # QUERY
def parse_query(file_path=os.path.join(DATASET_DIR, QUERY_FILENAME)):
    """
    Parse a query file in CISI format and extract queries with metadata.

    Args:
        file_path (str, optional): Path to the query file. Defaults to DATASET_DIR/QUERY_FILENAME.

    Returns:
        list[dict]: List of queries, each as a dictionary with keys:
            - "id" (int): Query ID.
            - "title" (str): Query title (if present).
            - "author" (str): Author name (if present).
            - "content" (str): Main query content.
            - "biblio" (str): Bibliographic info (if present).

    Example:
    ```
        [
            {
                "id": 1,
                "title": "Information Retrieval",
                "author": "J. Doe",
                "content": "What is information retrieval?",
                "biblio": "Journal of IR, 2020"
            },
            ...
        ]
    ```
    """
    queries = []
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split queries by .I marker
    query_blocks = re.split(r'^\s*\.I ', content, flags=re.MULTILINE)

    for block in query_blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        query_id = int(lines[0].strip())
        query = {
            "id": query_id,
            "title": "",
            "author": "",
            "content": "",
            "biblio": ""
        }
        body = '\n'.join(lines[1:]).strip()

        # Extract content (.W) - always present
        _content = re.search(r'\.W\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _content:
            query["content"] = _content.group(1).strip()

        # Extract optional fields
        _title = re.search(r'\.T\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _title:
            query["title"] = _title.group(1).strip()

        _author = re.search(r'\.A\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _author:
            query["author"] = _author.group(1).strip()

        _biblio = re.search(r'\.B\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _biblio:
            query["biblio"] = _biblio.group(1).strip()

        queries.append(query)

    return queries


# # CISI

def tokenize_nltk(text: str, remove_punctuation: bool = True) -> list[str]:
    """
    Tokenize input text using NLTK, convert to lowercase, and optionally remove punctuation.

    Args:
        text (str): The input text to tokenize.
        remove_punctuation (bool, optional): If True, remove punctuation from tokens. Defaults to True.

    Returns:
        list[str]: List of processed tokens.
        
    Example:
        tokenize_nltk("Hello, World!")  # ['hello', 'world']
    """
    # Convert text to lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())

    if remove_punctuation:
        # Remove punctuation and filter out empty tokens
        tokens = [word for word in (w.translate(str.maketrans('', '', string.punctuation)) for w in tokens) if word]

    return tokens

    

def _calculate_raw_tf(words: list[str]) -> dict[str, int]:
    """
    Calculates the raw term frequency (TF) of each word in a list.
    Args:
        words (list[str]): A list of words (tokens) for which to compute term frequencies.
    Returns:
        _ (dict[str,int]): A dictionary mapping each unique word to its frequency count in the input list.
    Example:
        >>> _calculate_raw_tf(['apple', 'banana', 'apple'])
        {'apple': 2, 'banana': 1}
    """
    
    
    tf = defaultdict(int)
    for word in words:
        tf[word] += 1
    
    return dict(tf)

def _calculate_idf(documents: list[tuple]) -> dict[str, float]:
    """
    Compute inverse document frequency (IDF) for each term in a corpus.

    Args:
        documents (list[tuple]): List of (doc id, tokenized documents), each as a tuple of (doc_id, terms).

    Returns:
        dict[str, float]: Mapping from term to its IDF score (log2(N/df)), where N is the number of documents and df is the document frequency of the term.

    Example:
        >>> _calculate_idf([(1, ["apple", "banana"]), (2, ["apple", "carrot"])])
        {'apple': 1.0, 'banana': 1.0, 'carrot': 1.0}
    """
    df = defaultdict(int)
    total_documents = len(documents)
    # Count document frequency for each term
    for doc_id, terms in documents:
        for term in set(terms):  # Count each term once per document
            df[term] += 1

    idf = {}
    for term, freq in df.items():
        # Avoid division by zero; freq should never be zero here
        idf[term] = math.log2(total_documents / freq) if freq else 0.0
    return idf

def _inverted_index_by_term(documents: list[tuple]) -> dict[str, list[int]]:
    """
    Build an inverted index mapping each term to the list of document IDs containing it.

    Args:
        documents (list[tuple]): List of (doc id, tokenized documents), each as a tuple of (doc_id, terms).

    Returns:
        dict[str, list[int]]: Dictionary where each key is a term and the value is a list of document IDs (int) in which the term appears.

    Example:
        >>> _inverted_index_by_term([(1, ["apple", "banana"]), (2, ["banana", "carrot"])])
        {'apple': [1], 'banana': [1, 2], 'carrot': [2]}
    """
    inverted_index = defaultdict(list)
    for doc_id, terms in documents:
        # Use a set to ensure each doc_id appears only once per term
        for term in set(terms):
            inverted_index[term].append(doc_id)
    return dict(inverted_index)

def _inverted_index_by_doc(documents: list[tuple]) -> dict[int, dict[str, list[int]]]:
    """
    Build a positional inverted index for each document.

    Args:
        documents (list[tuple]): List of (doc id, tokenized documents), each as a tuple of (doc_id, terms).

    Returns:
        dict[int, dict[str, list[int]]]: Dictionary mapping document IDs (int) to a dictionary,
            where each key is a term (str) and the value is a list of positions (int) where the term appears.

    Example:
        {
            0: {"information": [0, 4], "retrieval": [1]},
            1: {"information": [3]},
            2: {"retrieval": [5, 7]}
        }
    """
    # For each document, map each term to its positions within the document
    inverted_index = defaultdict(lambda: defaultdict(list))
    for doc_id, doc in documents:
        for position, term in enumerate(doc):
            inverted_index[doc_id][term].append(position)
    # Convert defaultdicts to dicts for output
    return {doc_id: dict(term_dict) for doc_id, term_dict in inverted_index.items()}
    

def parse_cisi_all(file_path=os.path.join(DATASET_DIR, CISI_ALL_FILENAME)):
    """
    Parses the CISI ALL dataset file and returns a dictionary containing processed documents, IDF values, and inverted indexes.
    Args:
        file_path (str, optional): Path to the CISI ALL dataset file. Defaults to os.path.join(DATASET_DIR, CISI_ALL_FILENAME).
    Returns:
        dict: A dictionary with the following structure:
            {
                "docs": List[dict],  # List of document dictionaries, each containing 'id', 'title', 'author', 'content', 'tokenized_content', and 'raw_tf'
                "idf": dict,         # Inverse document frequency for each term
                "inverted_index_by_term": dict,  # Mapping from term to list of document ids containing the term
                "inverted_index_by_doc": dict,   # Mapping from document id to list of terms in the document
            }
    
    Example:
        {
            "docs": [
                {
                    "id": 1,
                    "title": "Information Retrieval",
                    "author": "J. Doe",
                    "content": "What is information retrieval?",
                    "tokenized_content": ["what", "is", "information", "retrieval"],
                    "raw_tf": {"what": 1, "is": 1, "information": 1, "retrieval": 1}
                },
                ...
            ],
            "idf": {"information": 1.0, "retrieval": 1.0, ...},
            "inverted_index_by_term": {"information": [0], "retrieval": [0], ...},
            "inverted_index_by_doc": {0: {"information": [0,1], "retrieval": [3]}, ...}
        }
    """
    
    cisi_all = {
        "docs": [],
        "idf": {},
        "inverted_index_by_term": {},
        "inverted_index_by_doc": {},
        
    }
    
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Split by .I
    blocks = re.split(r'^\s*\.I ', content, flags=re.MULTILINE)
    
    docs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.splitlines()
        
        # Extract document ID from the first line of the block
        doc_id = int(lines[0].strip())
        doc = {
            "id": doc_id,
            "title": "",
            "author": "",
            "content": "",
            "tokenized_content": [],
            "raw_tf": {},
        }
        
        body = '\n'.join(lines[1:]).strip()
        
        # extract the title .T, author .A, and content .W
        _title = re.search(r'\.T\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _title:
            doc["title"] = _title.group(1).strip()
        
        _author = re.search(r'\.A\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _author:
            doc["author"] = _author.group(1).strip()
        
        _content = re.search(r'\.W\n(.*?)(\n\.|$)', body, re.DOTALL)
        if _content:
            doc["content"] = _content.group(1).strip()
        
        # Tokenize the content
        doc["tokenized_content"] = tokenize_nltk(doc["content"])    
        
        # Calculate raw term frequency
        words = doc["tokenized_content"]
        doc["raw_tf"] = _calculate_raw_tf(words)
        
        docs.append(doc)

    # Calculate IDF for all documents
    documents = [(doc["id"], doc["tokenized_content"]) for doc in docs]
    idf = _calculate_idf(documents)
    
    inverted_index_by_term = _inverted_index_by_term(documents)
    
    inverted_index_by_doc = _inverted_index_by_doc(documents)
    
    # Store the results in cisi_all
    cisi_all["docs"] = docs
    cisi_all["idf"] = idf
    cisi_all["inverted_index_by_term"] = inverted_index_by_term
    cisi_all["inverted_index_by_doc"] = inverted_index_by_doc
    
    return cisi_all


def save_as_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
def save_as_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Parse qrels
    qrels = parse_qrels()
    save_as_json(qrels, os.path.join(PREPROCESSED_DIR, "qrels.json"))
    
    # Parse query
    queries = parse_query()
    save_as_json(queries, os.path.join(PREPROCESSED_DIR, "queries.json"))
    
    # Parse cisi.all (to pickle for fast access)
    cisi_all = parse_cisi_all()

    save_as_json(cisi_all, os.path.join(PREPROCESSED_DIR, "cisi_all.json"))
    save_as_pickle(cisi_all, os.path.join(PREPROCESSED_DIR, "cisi_all.pkl"))
    
    print("Preprocessing completed successfully.")