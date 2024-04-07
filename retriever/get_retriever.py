from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from loguru import logger

from data_ingestion.nodes_processing import CodeNodes

from .embeddings import get_embeddings


def index_given_nodes(
    code_nodes: CodeNodes,
    embedding_model: MistralAIEmbedding | OpenAIEmbedding | FastEmbedEmbedding,
    hybrid_search: bool,
    recreate_collection: bool = False,
) -> VectorStoreIndex:
    """
    Given a list of nodes, create a new index or use an existing one.

    Parameters
    ----------
    code_nodes : CodeNodes
        The nodes to index.
    embedding_model : MistralAIEmbedding | OpenAIEmbedding | FastEmbedEmbedding
        The embedding model to use.
    hybrid_search : bool
        Whether to enable hybrid search.
    recreate_collection : bool, optional
        Whether to recreate the collection, by default False.

    """

    collection_name = code_nodes.nodes_config
    client = QdrantClient("localhost", port=6333)
    if recreate_collection:
        client.delete_collection(collection_name)

    try:
        count = client.count(collection_name).count
    except UnexpectedResponse:
        count = 0

    if count == len(code_nodes.nodes):
        logger.info(f"Found {count} existing nodes. Using the existing collection.")
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=client,
            enable_hybrid=hybrid_search,
        )
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embedding_model,
        )
    logger.info(
        f"Found {count} existing nodes. Creating a new index with {len(code_nodes.nodes)} nodes. This may take a while."
    )
    if count > 0:
        client.delete_collection(collection_name)

    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=client,
        enable_hybrid=hybrid_search,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        code_nodes.nodes,
        storage_context=storage_context,
        embed_model=embedding_model,
    )
    return index


def index_nodes(
    code_name: str,
    embedding_model: str,
    use_window_nodes: bool,
    nodes_window_size: int,
    hybrid_search: bool,
    recreate_collection: bool = False,
    reload_data: bool = False,
) -> tuple[VectorStoreIndex, list]:

    embed_model = get_embeddings(embedding_model)
    logger.info("Creating text nodes ...")
    code_nodes = CodeNodes(
        code_name=code_name,
        use_window_nodes=use_window_nodes,
        nodes_window_size=nodes_window_size,
        reload_data=reload_data,
    )
    if hybrid_search:
        code_nodes.nodes_config += "_hybrid"

    code_nodes.nodes_config += f"_{embedding_model.replace('/', '_')}"
    logger.info("Text nodes creation finished.")
    index = index_given_nodes(
        code_nodes, embed_model, hybrid_search, recreate_collection
    )
    return index, code_nodes.post_processors
