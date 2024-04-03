from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.tools import QueryEngineTool

from query.query_engine import create_query_engine


codes_to_description = {
    "Code civil": "Code civil: code juridique qui regroupe les lois relatives au droit civil français, c’est-à-dire l'ensemble des règles qui déterminent le statut des personnes (livre Ier), celui des biens (livre II) et celui des relations entre les personnes privées (livres III et IV).",
    "Code général des impôts": "Code général des impôts: code juridique qui regroupe les lois relatives aux impôts en France, c’est-à-dire l'ensemble des règles qui déterminent les impôts et les taxes.",
    "Code de la propriété intellectuelle": "Code de la propriété intellectuelle: code juridique qui regroupe les lois relatives à la propriété intellectuelle en France, c’est-à-dire l'ensemble des règles qui déterminent les droits des auteurs, des artistes-interprètes, des producteurs de phonogrammes et de vidéogrammes et des entreprises de communication audiovisuelle.",
    "Code de la route": "Code de la route: code juridique qui regroupe les lois relatives à la circulation routière en France, c’est-à-dire l'ensemble des règles qui déterminent les droits et les devoirs des usagers de la route.",
    "Code du travail": "Code du travail: code juridique qui regroupe les lois relatives au droit du travail en France, c’est-à-dire l'ensemble des règles qui déterminent les droits et les devoirs des employeurs et des salariés.",
}


def get_tools():
    tools = []
    for code_name, code_description in codes_to_description.items():
        query_engine = create_query_engine(code_name=code_name)
        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            description=code_description,
        )
        tools.append(tool)
    return tools


def create_routing_engine():
    query_engine_tools = get_tools()
    query_engine = RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(),
        query_engine_tools=query_engine_tools,
    )
    return query_engine
