from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import CompletionResponse
import re
import json

openai_kwargs = {
    "temperature": 0,
    "max_tokens": 4000,
    "verbose": True,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": -1,
}

CODE_NAME = "code penal"
PROMPT = f"Tu es un citoyen français qui a des questions par rapport au {CODE_NAME} français. Retourne une liste de questions spécifiques par rapport au {CODE_NAME} français. Tu dois retourner une liste avec {{n_qu}} questions différentes sous la forme d'une liste de strings. Dans les questions, ne mentionne pas le code civil et ne mets pas des préfixes type [q1] ou 1., pose directement la question."


def parse_response(resp: CompletionResponse):
    questions = resp.text.split("\n")
    # remove 1. 2. from beginning of questions
    questions = [json.loads(re.sub(r"^\d+\.\s", "", qu)) for qu in questions]
    return questions


def generate_questions(n_qu: int = 50):
    openai = OpenAI(model="gpt-4-0125-preview", **openai_kwargs)
    resp = openai.complete(PROMPT.format(n_qu=n_qu))
    questions = parse_response(resp)
    return questions


if __name__ == "__main__":
    questions = generate_questions()
    with open(f"./data/questions_{CODE_NAME.replace(' ', '_')}.json", "w") as js:
        json.dump(questions, js)
