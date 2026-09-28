import os
import httpx
from dotenv import load_dotenv

# Load environment variables FIRST before importing components that need them
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain.chat_models import init_chat_model

from src.rag_pipeline import RagPipeline
from evals.harness import load_ground_truth, summarize_by_metric, print_summary

APIMASTER_API_KEY = os.getenv("APIMASTER_API_KEY")
APIMASTER_BASE_URL = os.getenv("APIMASTER_BASE_URL")
MODEL_NAME = "gpt-5.6-luna"

llm = init_chat_model(
    MODEL_NAME,
    model_provider="openai",
    api_key=APIMASTER_API_KEY,
    base_url=APIMASTER_BASE_URL,
    http_client=httpx.Client(trust_env=False),
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

GT_PATH = "gt/correctness_gt.json"
JUDGE_MODEL = LangChainDeepEvalWrapper(llm)
THRESHOLD = 0.7


def run(rag):
    # 1. LOAD queries + ideal answers
    gt = load_ground_truth(GT_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in gt:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                expected_output=g["ideal_answer"],
            )
        )

    # 3. THREE APPLICATION-LEVEL QUALITY METRICS

    # 3a. CORRECTNESS --- reference-based, judges TRUTH (not coverage or length)
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare only the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
            "A factually accurate answer must score at least 0.9 even if it is shorter or covers fewer points than the expected output.",
            "Do NOT deduct for brevity, missing elaboration, or omitted points --- omissions are not errors here.",
            "Additional correct information must NEVER lower the score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent. No contradictions. Brevity is fine."),
            Rubric(score_range=(5, 8),  expected_outcome="Mostly correct but one minor inaccuracy."),
            Rubric(score_range=(0, 4),  expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # 3b. COMPLETENESS --- reference-based, judges COVERAGE (not correctness)
    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed in the actual output.",
            "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
            "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly --- factual correctness is judged separately.",
            "Do NOT penalize the actual output for adding extra information beyond the expected output.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
            Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
            Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # 3c. PROFESSIONALISM --- reference-free, judges TONE only
    professionalism = GEval(
        name="Professionalism",
        criteria="Assess the level of professionalism and expertise conveyed in the response.",
        evaluation_steps=[
            "Determine whether the actual output maintains a professional tone throughout.",
            "Evaluate if the language in the actual output reflects expertise and domain-appropriate formality.",
            "Ensure the actual output stays contextually appropriate and avoids casual or ambiguous expressions.",
            "Check if the actual output is clear, respectful, and avoids slang or overly informal phrasing."
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )

    # 4. EVALUATE --- all three together (added run_async=False to prevent timeout)
    result = evaluate(test_cases=test_cases, metrics=[correctness, completeness, professionalism])
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline(fetch_k=15, top_k=2))


if __name__ == "__main__":
    print_summary("application", run_local())