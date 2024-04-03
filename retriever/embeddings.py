import os

from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from schema import MistralSupportedModels, OpenAISupportedModels


def get_mistral_embeddings(
    embeddings_model_name: MistralSupportedModels = MistralSupportedModels.MISTRAL_EMBED,
):

    if not embeddings_model_name == "mistral-embed":
        raise ValueError(
            f"Embeddings model {embeddings_model_name} not supported."
            f"For now, only `mistral-embed` is supported for Mistral embeddings."
        )

    embed_model = MistralAIEmbedding(
        model_name=embeddings_model_name, api_key=os.getenv("MISTRAL_API_KEY")
    )
    return embed_model


def get_openai_embeddings(
    embeddings_model_name: OpenAISupportedModels = OpenAISupportedModels.ADA,
):

    embed_model = OpenAIEmbedding(model=embeddings_model_name)
    return embed_model


def get_fastembed_embeddings(
    embeddings_model_name: str = "fastembed",
):

    embed_model = FastEmbedEmbedding(model_name=embeddings_model_name)
    return embed_model


def get_embeddings(
    embeddings_model_name: str = "ada",
) -> MistralAIEmbedding | OpenAIEmbedding | FastEmbedEmbedding:
    if "mistral" in embeddings_model_name:
        model_name = MistralSupportedModels(embeddings_model_name)
        return get_mistral_embeddings(model_name)
    elif "text-embedding" in embeddings_model_name:
        model_name = OpenAISupportedModels(embeddings_model_name)
        return get_openai_embeddings(model_name)

    elif "multilingual" in embeddings_model_name:
        return get_fastembed_embeddings(embeddings_model_name)

    else:
        raise ValueError(
            f"Embeddings model {embeddings_model_name} not supported."
            f"Supported models include `mistral-embed` and `openai-embed` and `fastembed`."
        )
