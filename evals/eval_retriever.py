from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from src.reranker import RerankingRetriever
from evals.harness import load_ground_truth, summarize_by_metric, print_summary

import os
import httpx
from langchain.chat_models import init_chat_model

import re
load_dotenv()

APIMASTER_API_KEY=os.getenv("APIMASTER_API_KEY")
APIMASTER_BASE_URL=os.getenv("APIMASTER_BASE_URL")
MODEL_NAME = "gpt-5.6-luna"

llm = init_chat_model(
        MODEL_NAME,
        model_provider="openai",
        api_key=APIMASTER_API_KEY,
        base_url=APIMASTER_BASE_URL,
        http_client=httpx.Client(trust_env=False),
        # timeout=60,
    )

class LangChainDeepEvalWrapper(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def _clean_json(self, text: str) -> str:
        text = text.replace("\\'", "'")
        return text

    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt).content
        return self._clean_json(res)

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return self._clean_json(res.content)

    def get_model_name(self):
        return getattr(self.model, "model_name", "custom_langchain_model")

GT_PATH = "gt/retriever_gt.json"
JUDGE_MODEL = LangChainDeepEvalWrapper(llm)
THRESHOLD = 0.7


def run(retriever):
    # 1. LOAD the gt set --- the fixed, human-authored truth
    gt = load_ground_truth(GT_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per gt.
    test_cases = []
    for g in gt:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True, async_mode=False),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "reranker",
            "embedding_model": "nvidia/llama-nemotron-embed-vl-1b-v2",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 6,
            "fetch_k": 15,
            "reranker": "BAAI/bge-reranker-v2-m3",
            "judge_model": JUDGE_MODEL.get_model_name(),
            "ground_truth": GT_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(RerankingRetriever(fetch_k=15, top_k=5))


if __name__ == "__main__":
    print_summary("retriever", run_local())