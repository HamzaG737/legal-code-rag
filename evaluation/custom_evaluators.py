from typing import Any, Optional, Sequence

from llama_index.core.evaluation import (
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
)
from llama_index.core.response import Response
from llama_index.core.schema import MetadataMode
from llama_index.core.evaluation.base import EvaluationResult


class CustomContextRelevancyEvaluator(ContextRelevancyEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        metadata_mode: MetadataMode = MetadataMode.ALL,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string and generated Response object.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        response_str: Optional[str] = None
        contexts: Optional[Sequence[str]] = None
        if response is not None:
            response_str = response.response
            contexts = [
                node.get_content(metadata_mode=metadata_mode)
                for node in response.source_nodes
            ]

        return self.evaluate(
            query=query, response=response_str, contexts=contexts, **kwargs
        )


class CustomFaithfulnessEvaluator(FaithfulnessEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        metadata_mode: MetadataMode = MetadataMode.ALL,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string and generated Response object.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        response_str: Optional[str] = None
        contexts: Optional[Sequence[str]] = None
        if response is not None:
            response_str = response.response
            contexts = [
                node.get_content(metadata_mode=metadata_mode)
                for node in response.source_nodes
            ]

        return self.evaluate(
            query=query, response=response_str, contexts=contexts, **kwargs
        )


class CustomAnswerRelevancyEvaluator(AnswerRelevancyEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        metadata_mode: MetadataMode = MetadataMode.ALL,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string and generated Response object.

        Subclasses can override this method to provide custom evaluation logic and
        take in additional arguments.
        """
        response_str: Optional[str] = None
        contexts: Optional[Sequence[str]] = None
        if response is not None:
            response_str = response.response
            contexts = [
                node.get_content(metadata_mode=metadata_mode)
                for node in response.source_nodes
            ]

        return self.evaluate(
            query=query, response=response_str, contexts=contexts, **kwargs
        )
