import re
import os
from typing import List

from pylegifrance import recherche_CODE, LegiHandler
from loguru import logger

from utils import save_json
from .constants import possible_headers, path_dir_data


client = LegiHandler()
client.set_api_keys(
    legifrance_api_key=os.getenv("LEGIFRANCE_API_KEY"),
    legifrance_api_secret=os.getenv("LEGIFRANCE_API_SECRET"),
)


def get_all_articles(dict_code: dict) -> List[dict]:

    queue = [dict_code]
    articles = []
    while queue:
        current = queue.pop()
        articles += [
            {k: article[k] for k in ["pathTitle", "content", "num"]}
            for article in current["articles"]
            if article["etat"] == "VIGUEUR"
        ]
        for child in current["sections"]:
            queue.append(child)
    return articles


def clean_content(content: str) -> str:
    # Remove the <a> tags but keep their text content
    no_links_text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", content)

    # Split the text into paragraphs and filter out empty ones
    paragraphs = re.split(r"</?p>", no_links_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip() != ""]

    # Join the paragraphs separated by \n
    clean_text = "\n".join(paragraphs)

    return clean_text


def get_different_headers(path_title: List[str]) -> dict:
    cleaned_path_title = [tit.split(":", 1) for tit in path_title]
    dict_output = {}
    for info in cleaned_path_title:
        if len(info) < 2:
            logger.debug(f"Skipped header : {info}")
            continue
        header_name, header_val = info
        lowered_and_stripped_header = header_name.lower().strip()
        for possible_header in possible_headers:
            if lowered_and_stripped_header.startswith(possible_header):
                dict_output[possible_header] = header_val.strip()
                break
    return dict_output


def parse_row(row: dict) -> dict:
    return {
        "content": clean_content(row["content"]),
        "num": row["num"],
        **get_different_headers(path_title=row["pathTitle"]),
    }


def get_code_articles(
    code_name: str = "Code civil", save_rows: bool = True
) -> List[dict]:
    dict_code = recherche_CODE(code_name, champ="ALL")[0]
    articles = get_all_articles(dict_code=dict_code)
    preprocessed_articles = list(map(parse_row, articles))
    no_duplicates_rows = list({r["num"]: r for r in preprocessed_articles}.values())

    os.makedirs(path_dir_data, exist_ok=True)

    path_data = os.path.join(path_dir_data, f"{code_name}.json")
    if save_rows:
        save_json(no_duplicates_rows, path_data)
    return no_duplicates_rows
