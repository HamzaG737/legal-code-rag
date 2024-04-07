from llama_index.core.query_engine import BaseQueryEngine, RetrieverQueryEngine
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever

from retriever.get_retriever import index_nodes
from query.constants import QUERY_GEN_PROMPT


def update_prompts_for_query_engine(query_engine: BaseQueryEngine) -> BaseQueryEngine:

    new_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "You must always mention the code name and article number in your answer. With something like 'selon le code x et l'article y, ...'\n"
        "You must also detect the language of the query and translate your response to the original query language. \n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_tmpl = PromptTemplate(new_tmpl_str)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": new_tmpl})
    return query_engine


def get_query_fusion_retrieval(
    index: VectorStoreIndex,
    postprocessors_list: list,
    similarity_top_k: int = 5,
    sparse_top_k: int = 0,
    hybrid_search_alpha: float = 0.5,
    hybrid_search: bool = False,
    num_generated_questions: int = 4,
) -> BaseQueryEngine:
    kwargs = {"similarity_top_k": similarity_top_k}
    if hybrid_search:
        kwargs.update(
            {
                "vector_store_query_mode": "hybrid",
                "sparse_top_k": sparse_top_k,
                "alpha": hybrid_search_alpha,
            }
        )

    retriever = index.as_retriever(**kwargs)

    retriever = QueryFusionRetriever(
        [retriever],
        similarity_top_k=similarity_top_k,
        num_queries=num_generated_questions,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
        query_gen_prompt=QUERY_GEN_PROMPT,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever, node_postprocessors=postprocessors_list
    )
    query_engine = update_prompts_for_query_engine(query_engine)
    return query_engine


def get_query_engine_based_on_index(
    index: VectorStoreIndex,
    postprocessors_list: list,
    similarity_top_k: int = 5,
    sparse_top_k: int = 0,
    hybrid_search_alpha: float = 0.5,
    hybrid_search: bool = False,
) -> BaseQueryEngine:
    kwargs = {
        "node_postprocessors": postprocessors_list,
        "similarity_top_k": similarity_top_k,
    }
    if hybrid_search:
        kwargs.update(
            {
                "vector_store_query_mode": "hybrid",
                "sparse_top_k": sparse_top_k,
                "alpha": hybrid_search_alpha,
            }
        )
    query_engine = index.as_query_engine(**kwargs)
    query_engine = update_prompts_for_query_engine(query_engine)
    return query_engine


def create_query_engine(
    code_name: str = "Code civil",
    embedding_model: str = "text-embedding-ada-002",
    similarity_top_k: int = 5,
    sparse_top_k: int = 0,
    hybrid_search: bool = False,
    hybrid_search_alpha: float = 0.5,
    use_window_nodes: bool = False,
    query_rewrite: bool = False,
    num_generated_questions: int = 4,
    nodes_window_size: int = 3,
    recreate_collection: bool = False,
    reload_data: bool = False,
) -> BaseQueryEngine:
    """
    Create a Llama index query engine with the given configuration.

    Parameters
    ----------

    code_name : str, optional
        The name of the code to index, by default "code_civil"

    embedding_model : str, optional
        The name of the embedding model in the retriever, by default "text-embedding-ada-002"

    similarity_top_k : int, optional
        The number of top similar nodes to return, by default 5

    use_relations : bool, optional
        Whether to include relationships between nodes, by default False

    use_window_nodes : bool, optional
        Whether to include window nodes, by default False

    query_rewrite : bool, optional
        Whether to use query rewrite, by default False

    num_generated_questions : int, optional
        The number of generated questions by question for the query rewrite, by default 4.
        This option is only used when query_rewrite is set to True.


    nodes_window_size : int, optional
        The size of the window nodes, by default 3

    recreate_collection : bool, optional
        Whether to recreate the database collection, by default False

    reload_data : bool, optional
        Whether to reload the data from the legifrance API, by default False

    """
    index, postprocessors_list = index_nodes(
        code_name=code_name,
        embedding_model=embedding_model,
        use_window_nodes=use_window_nodes,
        nodes_window_size=nodes_window_size,
        hybrid_search=hybrid_search,
        recreate_collection=recreate_collection,
        reload_data=reload_data,
    )
    kwargs_query_engine = {
        "index": index,
        "postprocessors_list": postprocessors_list,
        "similarity_top_k": similarity_top_k,
        "sparse_top_k": sparse_top_k,
        "hybrid_search_alpha": hybrid_search_alpha,
        "hybrid_search": hybrid_search,
    }
    if query_rewrite:
        return get_query_fusion_retrieval(
            **kwargs_query_engine, num_generated_questions=num_generated_questions
        )

    return get_query_engine_based_on_index(**kwargs_query_engine)
