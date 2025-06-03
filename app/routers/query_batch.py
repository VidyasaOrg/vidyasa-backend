from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from app.utils.preprocessing import tokenize_words, preprocess_text
import tempfile
import os
import json

from app.services import get_qrels, get_queries, get_irdata
from app.schemas import QueryResponse, DocumentSimilarityScore, TermFrequencyMethod
from app.models import Qrels
from app.models import Query

from app.services.query_expansion import expand_query_kb

# Import kelas-kelas yang diperlukan
from app.services.coefficient_calculator import CoefficientCalculator
from app.services.map_calculator import MAPCalculator

router = APIRouter(prefix="/query_batch", tags=["batch_query"])

def calculate_similarity_ranking(query: Query, docs, coefficient_calc: CoefficientCalculator, 
                               term_doc_matrix, doc_vectors, tf_method='raw', 
                               use_idf=True, use_cosine_norm=False) -> list[DocumentSimilarityScore]:
    """
    Menghitung ranking dokumen berdasarkan similarity dengan query
    """
    # Process query menjadi vector
    query_vector = coefficient_calc.process_query(
        query.content, 
        term_doc_matrix,
        do_stemming=True,
        remove_stopwords=True,
        tf_method=tf_method,
        use_idf=use_idf,
        use_cosine_norm=use_cosine_norm
    )
    
    # Hitung similarity dengan setiap dokumen
    similarities = []
    for doc_id, doc_vector in doc_vectors.items():
        similarity = coefficient_calc.calculate_cosine_similarity(query_vector, doc_vector)
        similarities.append(DocumentSimilarityScore(doc_id=doc_id, similarity_score=similarity))
    
    # Sort berdasarkan similarity score (descending)
    similarities.sort(key=lambda x: x.similarity_score, reverse=True)
    
    return similarities[:10]  # Return top 10

def calculate_query_weights(query: str, term_doc_matrix, coefficient_calc: CoefficientCalculator,
                          tf_method='raw', use_idf=True, use_cosine_norm=False) -> dict:
    """
    Menghitung bobot untuk setiap term dalam query
    """
    query_vector = coefficient_calc.process_query(
        query, 
        term_doc_matrix,
        do_stemming=True,
        remove_stopwords=True,
        tf_method=tf_method,
        use_idf=use_idf,
        use_cosine_norm=use_cosine_norm
    )
    
    return query_vector

def expand_query_advanced(query: str, relevant_docs_texts: list, expansion_terms_count):
    """
    Expand query menggunakan knowledge base dari dokumen relevan
    """
    if expansion_terms_count == 0:
        return query
    
    # Gunakan fungsi expand_query_kb yang sudah ada
    expansion_result = expand_query_kb(query, relevant_docs_texts)
    
    if expansion_terms_count == "all":
        expanded_terms = list(expansion_result.keys())
    else:
        # Ambil top-k terms berdasarkan weight
        sorted_terms = sorted(expansion_result.items(), key=lambda x: x[1], reverse=True)
        expanded_terms = [term for term, weight in sorted_terms[:expansion_terms_count]]
    
    # Gabungkan query asli dengan expanded terms
    original_terms = query.split()
    all_terms = original_terms + [term for term in expanded_terms if term not in original_terms]
    
    return " ".join(all_terms)

def calculate_map_score(ranking_ids: list[int], relevant_docs_set: set[int], 
                       map_calculator: MAPCalculator) -> float:
    """
    Menghitung MAP score untuk ranking hasil
    """
    if not ranking_ids or not relevant_docs_set:
        return 0.0
    
    # Convert set ke list untuk kompatibilitas
    relevant_docs_list = list(relevant_docs_set)
    
    # Hitung Average Precision
    ap = map_calculator.calculate_average_precision(ranking_ids, relevant_docs_list)
    return ap

@router.post("/", response_class=FileResponse)
async def search_batch_queries(
    file: UploadFile = File(..., description="File containing queries (one per line or JSON format)"),
    is_stemming: bool = Form(False, description="Apply stemming to queries"),
    is_stop_words_removal: bool = Form(False, description="Remove stop words from queries"),
    term_frequency_method: TermFrequencyMethod = Form(TermFrequencyMethod.RAW, description="Term frequency method to use"),
    expansion_terms_count: int = Form("all", description="Number of terms to expand the query with (0 for no expansion, 'all' for all terms)")
):
    """
    Process a batch of queries from a file and return the results as a JSON file.
    The file can contain queries in JSON format or as newline-separated raw queries.
    Each query will be processed to retrieve original and expanded rankings, MAP scores, and term weights.
    
    Example of file content:
    ```
    [
        {"query": "information retrieval"},
        {"query": "database systems"},
        {"query": "machine learning"}
    ]
    ```
    
    Example request body:
    ```json
    {
        "file": "<file with queries>",
        "is_stemming": true,
        "is_stop_words_removal": false,
        "term_frequency_method": "log",
        "expansion_terms_count": 5
    }
    ```
    """
    try:
        contents = await file.read()
        
        # Parse queries dari file
        try:
            # Coba parse sebagai JSON dulu
            queries_data = json.loads(contents.decode())
            if not isinstance(queries_data, list):
                queries_data = [{"query": contents.decode().strip()}]
        except json.JSONDecodeError:
            # Jika bukan JSON, parse sebagai newline-separated queries
            queries_data = [{"query": line.strip()} for line in contents.decode().splitlines() if line.strip()]

        # Initialize calculators
        coefficient_calc = CoefficientCalculator()
        map_calculator = MAPCalculator()
        
        # Get data
        qrels: Qrels = get_qrels()
        irdata = get_irdata()
        docs = irdata

        # Build term-document matrix dari semua dokumen
        document_texts = [doc.content for doc in docs.documents]
        
        # Map term frequency method
        tf_method_map = {
            TermFrequencyMethod.RAW: 'raw',
            TermFrequencyMethod.LOG: 'logarithmic',
            TermFrequencyMethod.BINARY: 'binary',
            TermFrequencyMethod.AUGMENTED: 'augmented'
        }
        tf_method = tf_method_map.get(term_frequency_method, 'raw')
        
        # Build term-document matrix
        term_doc_matrix, doc_vectors, vocabulary = coefficient_calc.build_term_document_matrix(
            document_texts,
            do_stemming=is_stemming,
            remove_stopwords=is_stop_words_removal,
            tf_method=tf_method,
            use_idf=True,
            use_cosine_norm=False
        )

        # Process each query
        responses = []
        for idx, query_entry in enumerate(queries_data):
            raw_query = query_entry["query"]
            
            # Preprocess query
            tokens = tokenize_words(raw_query)
            processed_tokens = preprocess_text(tokens, is_stem=is_stemming, remove_stop_words=is_stop_words_removal)
            processed_query = " ".join(processed_tokens)

            query_obj = Query(id=idx, content=processed_query)

            # Get relevant documents for this query (if available in qrels)
            relevant_docs_set = set(qrels.get_relevant_docs(idx)) if idx in qrels.data else set()

            # ORIGINAL QUERY PROCESSING
            original_ranking_objs = calculate_similarity_ranking(
                query_obj, docs, coefficient_calc, term_doc_matrix, doc_vectors,
                tf_method=tf_method, use_idf=True, use_cosine_norm=False
            )
            original_ranking_ids = [doc.doc_id for doc in original_ranking_objs]
            original_map = calculate_map_score(original_ranking_ids, relevant_docs_set, map_calculator)
            original_query_weights = calculate_query_weights(
                processed_query, term_doc_matrix, coefficient_calc,
                tf_method=tf_method, use_idf=True, use_cosine_norm=False
            )

            # EXPANDED QUERY PROCESSING
            # Get top documents for expansion (use top 5 documents from original ranking)
            top_docs_for_expansion = original_ranking_ids[:5]
            relevant_docs_texts = [doc.content for doc in docs.documents if doc.id in top_docs_for_expansion]
            
            # Expand query
            expanded_query_text = expand_query_advanced(processed_query, relevant_docs_texts, expansion_terms_count)

            # Preprocess expanded query
            expanded_tokens = tokenize_words(expanded_query_text)
            expanded_processed_tokens = preprocess_text(expanded_tokens, is_stem=is_stemming,
                                                        remove_stop_words=is_stop_words_removal)
            expanded_processed_query = " ".join(expanded_processed_tokens)

            # Calculate ranking for expanded query
            expanded_query_obj = Query(id=idx, content=expanded_processed_query)
            expanded_ranking_objs = calculate_similarity_ranking(
                expanded_query_obj, docs, coefficient_calc, term_doc_matrix, doc_vectors,
                tf_method=tf_method, use_idf=True, use_cosine_norm=False
            )
            expanded_ranking_ids = [doc.doc_id for doc in expanded_ranking_objs]
            expanded_map = calculate_map_score(expanded_ranking_ids, relevant_docs_set, map_calculator)
            expanded_query_weights = calculate_query_weights(
                expanded_processed_query, term_doc_matrix, coefficient_calc,
                tf_method=tf_method, use_idf=True, use_cosine_norm=False
            )

            # Create response
            response = QueryResponse(
                original_query=raw_query,
                original_ranking=original_ranking_objs,
                original_map_score=original_map,
                original_query_weights=original_query_weights,
                expanded_query=expanded_query_text,
                expanded_ranking=expanded_ranking_objs,
                expanded_map_score=expanded_map,
                expanded_query_weights=expanded_query_weights
            )
            responses.append(response.model_dump())

        # Save to temporary file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
        json.dump(responses, temp, indent=2, ensure_ascii=False)
        temp.close()

        return FileResponse(temp.name, filename="batch_query_results.json", media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Additional helper endpoint untuk debugging
@router.get("/test_similarity")
async def test_similarity_calculation():
    """
    Endpoint untuk testing perhitungan similarity
    """
    try:
        coefficient_calc = CoefficientCalculator()
        
        # Test documents
        test_docs = [
            "information retrieval system",
            "database management system", 
            "machine learning algorithm"
        ]
        
        # Build matrix
        term_doc_matrix, doc_vectors, vocabulary = coefficient_calc.build_term_document_matrix(
            test_docs, tf_method='raw', use_idf=True
        )
        
        # Test query
        test_query = "information system"
        query_vector = coefficient_calc.process_query(test_query, term_doc_matrix)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_vector in doc_vectors.items():
            sim = coefficient_calc.calculate_cosine_similarity(query_vector, doc_vector)
            similarities.append({"doc_id": doc_id, "similarity": sim})
        
        return {
            "vocabulary": vocabulary,
            "query_vector": query_vector,
            "similarities": similarities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {str(e)}")