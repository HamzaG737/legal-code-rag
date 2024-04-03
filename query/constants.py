QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query. The queries must be in French and specifically "
    "adapted to query the French Civil Code, addressing the ambiguities in the input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
