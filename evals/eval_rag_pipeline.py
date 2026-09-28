import os
import httpx
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain.chat_models import init_chat_model

from src.rag_pipeline import RagPipeline
from evals.harness import load_ground_truth, summarize_by_metric, print_summary

load_dotenv()

APIMASTER_API_KEY = os.getenv("APIMASTER_API_KEY")
APIMASTER_BASE_URL = os.getenv("APIMASTER_BASE_URL")
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

    def generate(self, prompt: str) -> str:
        res = self.model.invoke(prompt)
        return res.content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return getattr(self.model, "model_name", "custom_langchain_model")

GT_PATH = "gt/faithfulness_gt.json"   # reuse the queries
JUDGE_MODEL = LangChainDeepEvalWrapper(llm)
THRESHOLD = 0.7


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    gt = load_ground_truth(GT_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in gt:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases, 
        metrics=metrics,
        hyperparameters={
            "judge_model": JUDGE_MODEL.get_model_name(),
            "ground_truth": GT_PATH,
        }
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline(fetch_k=8, top_k=3))


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())