from typing import List
from llama_index.core.schema import TextNode
from tqdm import tqdm

from .constants import possible_headers

WINDOW_METADATA_KEY = "window"
ORIGINAL_TEXT_METADATA_KEY = "original_text"


def add_window_nodes(nodes: List[TextNode], window_size: int = 3):
    """ """
    for i, node in tqdm(
        enumerate(nodes), total=len(nodes), desc="Adding window nodes ..."
    ):
        window_nodes = nodes[
            max(0, i - window_size) : min(i + window_size + 1, len(nodes))
        ]

        node.metadata[WINDOW_METADATA_KEY] = "\n".join(
            [n.get_content("llm") for n in window_nodes]
        )
        node.metadata[ORIGINAL_TEXT_METADATA_KEY] = node.text

        # exclude window metadata from embed and llm
        node.excluded_embed_metadata_keys.extend(
            [WINDOW_METADATA_KEY, ORIGINAL_TEXT_METADATA_KEY]
        )

        node.excluded_llm_metadata_keys.extend(
            [WINDOW_METADATA_KEY, ORIGINAL_TEXT_METADATA_KEY]
        )

    # since articles metadata (like title, chapter, etc ...) will be incorporated in WINDOW_METADATA_KEY,
    # we can exclude them from the llm metadata.

    for node in nodes:
        node.excluded_llm_metadata_keys.extend(possible_headers)

    return nodes
