from dataclasses import dataclass
from typing import List
import os
from uuid import uuid4
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

from loguru import logger

from utils import load_json
from .preprocess_legifrance_data import get_code_articles
from .constants import path_dir_data
from .window_nodes import add_window_nodes


@dataclass
class CodeNodes:
    code_name: str
    use_window_nodes: bool
    nodes_window_size: int = 3
    max_words_per_node: int = 2000
    _n_truncated_articles: int = 0

    def __post_init__(self):
        self.articles = self.try_load_data()
        self.nodes = self.create_nodes(self.articles)
        code_name_no_spaces = self.code_name.replace(" ", "_")
        self.nodes_config = f"{code_name_no_spaces}_base"
        self.post_processors = []
        if self.use_window_nodes:
            logger.info("Adding window nodes ...")
            self.nodes = add_window_nodes(self.nodes, self.nodes_window_size)
            self.nodes_config = f"{code_name_no_spaces}_window"
            self.post_processors.append(
                MetadataReplacementPostProcessor(target_metadata_key="window")
            )

    def create_nodes(self, list_articles: List[dict]):
        nodes = [
            TextNode(
                text=article["content"],
                id_=str(uuid4()),
                metadata=self._parse_metadata(article),
            )
            for article in list_articles
        ]

        return nodes

    def try_load_data(self) -> List[dict]:
        path = os.path.join(path_dir_data, f"{self.code_name}.json")
        try:
            code_articles = load_json(path=path)
        except FileNotFoundError:
            logger.warning(
                f"File not found at path {path}. Fetching data from Legifrance."
            )
            code_articles = get_code_articles(code_name=self.code_name)
        truncated_articles = self._chunk_long_articles(code_articles)
        return truncated_articles

    def _parse_metadata(self, article: dict) -> dict:
        metadata = {k: v for k, v in article.items() if k not in ["content", "num"]}
        metadata = {
            "Nom du code": self.code_name,
            **metadata,
            "Article numero": article["num"],
        }
        return metadata

    def _chunk_one_article(self, content: str) -> List[str]:
        words = content.split()
        chunks = []
        chunk = ""
        for word in words:
            if len(chunk) + len(word) > self.max_words_per_node:
                chunks.append(chunk)
                chunk = ""
            chunk += word + " "
        chunks.append(chunk)
        return chunks

    def _chunk_long_articles(self, articles: List[dict]) -> List[dict]:
        truncated_articles = []
        for article in articles:
            words = article["content"].split()
            if len(words) > self.max_words_per_node:
                self._n_truncated_articles += 1
                chunks = self._chunk_one_article(article["content"])
                for i, chunk in enumerate(chunks):
                    truncated_articles.append(
                        {
                            "content": chunk,
                            "id": article["num"] + f"_chunk_{i}",
                            **{k: v for k, v in article.items() if k != "content"},
                        }
                    )

            else:
                article["id"] = article["num"]
                truncated_articles.append(article)
        logger.info(
            f"Truncated {self._n_truncated_articles} articles into smaller chunks."
        )
        return truncated_articles
