import json
from typing import List, Dict

def format_dict_query(original_ranked_documents,
                expanded_ranked_documents,
                original_similarity_coefficient,
                expanded_similarity_coefficient,
                original_map,
                original_query,
                original_query_weights,
                expanded_map,
                expanded_query,
                expanded_query_weights):
    output = {
        "original-ranked-documents": original_ranked_documents,
        "expanded-ranked-documents": expanded_ranked_documents,
        "original-similarity-coefficient": original_similarity_coefficient,
        "expanded-similarity-coefficient": expanded_similarity_coefficient,
        "original-map": original_map,
        "original-query": original_query,
        "original-query-weights": original_query_weights,
        "expanded-map": expanded_map,
        "expanded-query": expanded_query,
        "expanded-query-weights": expanded_query_weights
    }
    return output


def format_and_save_results(
    results: List[Dict],
    output_file: str = "formatted_output.json"
):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)