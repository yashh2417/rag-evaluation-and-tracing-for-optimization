"""
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault --- the context was already correct.

    python -m evals.eval_generator
"""

import os
import httpx
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain.chat_models import init_chat_model

from src.generator import generate   
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

GT_PATH = "gt/faithfulness_gt.json"
JUDGE_MODEL = LangChainDeepEvalWrapper(llm)
THRESHOLD = 0.7


def run():
    # 1. LOAD the ground truth set (query + ideal_answer)
    gt = load_ground_truth(GT_PATH)

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in gt:
        # retrieval_context must be a list of strings
        context = g["ideal_context"]             
        answer = generate(g["query"], context)    

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=answer,             
                retrieval_context=context,        
            )
        )

    # 3. THE METRICS 
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,   
            async_mode=False
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
            async_mode=False
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(
        test_cases=test_cases, 
        metrics=metrics,
        hyperparameters={
            "judge_model": JUDGE_MODEL.get_model_name(),
            "ground_truth": GT_PATH,
        }
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())