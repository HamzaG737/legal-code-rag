from enum import Enum
from typing import Literal, List, Dict
import os

from IPython.display import display
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation.notebook_utils import get_eval_results_df
from llama_index.core.evaluation.base import EvaluationResult

from llama_index.core.schema import MetadataMode
import pandas as pd
from tqdm import tqdm
from loguru import logger

import evaluation.custom_evaluators as evals
from utils import load_json


PATH_SAVE_EVAL_METRICS = "./data/evaluation_results/{exp_name}_{code_name}_metrics.csv"
PATH_SAVE_EVAL_DEEP = "./data/evaluation_results/{exp_name}_{code_name}_deep.csv"

if not os.path.exists("./data/evaluation_results"):
    os.makedirs("./data/evaluation_results")

PATH_EVAL_CODE_CIVIL = "./data/questions_code_civil.json"
PATH_EVAL_CODE_CIVIL_QUERY_REWRITE = "./data/questions_code_civil_query_rewrite.json"
PATH_EVAL_CODE_DE_LA_ROUTE = "./data/questions_code_de_la_route.json"


class Metric(Enum):
    CONTEXT_RELEVANCY = "ContextRelevancy"
    ANSWER_RELEVANCY = "AnswerRelevancy"
    FAITHFULNESS = "Faithfulness"


def displayify_df(df):
    """For pretty displaying DataFrame in a notebook."""
    display_df = df.style.set_properties(
        **{
            "inline-size": "300px",
            "overflow-wrap": "break-word",
        }
    )
    display(display_df)


def evaluate_one_metric(
    query_engine: BaseQueryEngine,
    metric: Metric,
    code_name: Literal["code_civil", "code_de_la_route"],
    metadata_mode: MetadataMode,
    llm_for_eval: str = "gpt-3.5-turbo",
    if_query_rewrite: bool = False,
) -> List[EvaluationResult]:

    evaluator_class = getattr(evals, f"Custom{metric.value}Evaluator")

    evaluator = evaluator_class(llm=OpenAI(temperature=0, model=llm_for_eval))

    path_eval_data = f"PATH_EVAL_{code_name.upper()}"
    if if_query_rewrite:
        path_eval_data = f"PATH_EVAL_{code_name.upper()}_QUERY_REWRITE"
    eval_data = load_json(globals()[path_eval_data])

    results = []
    for question in tqdm(eval_data):
        response = query_engine.query(question)
        eval_result = evaluator.evaluate_response(
            query=question, response=response, metadata_mode=metadata_mode
        )

        results.append(eval_result)

    return results


def postprocess_eval_results(metric_to_results: Dict[str, pd.DataFrame]):

    mean_scores_df = pd.concat(
        [mdf.reset_index() for _, mdf in metric_to_results.items()],
        axis=0,
        ignore_index=True,
    )
    mean_scores_df = mean_scores_df.set_index("index")
    mean_scores_df.index = mean_scores_df.index.set_names(["metrics"])

    return mean_scores_df


def evaluate(
    query_engine: BaseQueryEngine,
    list_metrics: List[str],
    exp_name: str,
    code_name: Literal["code_civil", "code_de_la_route"],
    llm_for_eval: str = "gpt-3.5-turbo",
    do_save: bool = True,
    metadata_mode: MetadataMode = MetadataMode.ALL,
    if_query_rewrite: bool = False,
):
    deep_dfs = []
    mean_dfs = {}
    for metric_name in list_metrics:
        metric = Metric[metric_name.upper()]
        if "window" in exp_name.lower():
            metadata_mode = MetadataMode.NONE

        results = evaluate_one_metric(
            query_engine=query_engine,
            metric=metric,
            code_name=code_name,
            llm_for_eval=llm_for_eval,
            metadata_mode=metadata_mode,
            if_query_rewrite=if_query_rewrite,
        )
        deep_df, mean_df = get_eval_results_df(
            names=[exp_name] * len(results),
            results_arr=results,
            metric=metric_name,
        )
        deep_df["metric"] = metric_name
        deep_dfs.append(deep_df)
        mean_dfs[metric_name] = mean_df

    mean_scores_df = postprocess_eval_results(mean_dfs)
    deep_dfs_full = pd.concat(deep_dfs, axis=0, ignore_index=True)

    if do_save:
        mean_scores_df.to_csv(
            PATH_SAVE_EVAL_METRICS.format(exp_name=exp_name, code_name=code_name)
        )
        deep_dfs_full.to_csv(
            PATH_SAVE_EVAL_DEEP.format(exp_name=exp_name, code_name=code_name),
            index=False,
        )
    return mean_scores_df, deep_dfs_full, deep_dfs


def evaluate_multiple_experiments(
    general_exp_name: str,
    experiment_to_query_engine: Dict[str, BaseQueryEngine],
    list_metrics: List[str],
    code_name: Literal["code_civil", "code_de_la_route"],
    llm_for_eval: str = "gpt-3.5-turbo",
    do_save: bool = True,
    metadata_mode: MetadataMode = MetadataMode.ALL,
):
    dfs_mean_scores = []
    dfs_deep = []
    if_query_rewrite = "rewrit" in general_exp_name.lower()
    for exp_name, query_engine in experiment_to_query_engine.items():
        logger.info(f"Evaluating experiment: {exp_name}")
        mean_scores_df, deep_dfs_full, _ = evaluate(
            query_engine=query_engine,
            list_metrics=list_metrics,
            exp_name=exp_name,
            code_name=code_name,
            llm_for_eval=llm_for_eval,
            metadata_mode=metadata_mode,
            if_query_rewrite=if_query_rewrite,
        )
        dfs_mean_scores.append(mean_scores_df)
        dfs_deep.append(deep_dfs_full)

    dfs_mean_scores = pd.concat(dfs_mean_scores, axis=1)
    dfs_deep = pd.concat(dfs_deep, axis=0)

    if do_save:
        dfs_mean_scores.to_csv(
            PATH_SAVE_EVAL_METRICS.format(
                exp_name=general_exp_name, code_name=code_name
            )
        )
        dfs_deep.to_csv(
            PATH_SAVE_EVAL_DEEP.format(exp_name=general_exp_name, code_name=code_name),
            index=False,
        )

    return dfs_mean_scores, dfs_deep
