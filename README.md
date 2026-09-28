# RAG Evaluation & Tracing for Optimization

A comprehensive evaluation framework for the Retrieval-Augmented Generation (RAG) pipeline, covering both **Offline Regression Testing** and **Online Production Tracing**.

## Setup Instructions

1. **Environment Setup**  
   Ensure you are using Python 3.12+ and have your virtual environment activated:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**  
   Create a `.env` file in the root directory and define the API keys for the pipeline's components:
   ```bash
   # LangSmith Tracing
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT="Rag Evals"
   LANGSMITH_API_KEY="your_langsmith_key"
   
   # Judge Model
   JUDGE_LLM_API_KEY="your_judge_api_key"
   JUDGE_BASE_URL="your_judge_base_url"
   
   # Generator Model
   GENERATOR_LLM_API_KEY="your_generator_api_key"
   
   # Retriever (Embeddings)
   RETRIEVER_API_KEY="your_retriever_api_key"
   
   # Reranker
   RERANKER_API_KEY="your_reranker_api_key"
   ```

---

## Current Project Configuration & Tech Stack

This project is currently configured to use the following specific models and tools for the pipeline and evaluation:

- **Generator Model:** `gpt-5.6-terra` (via APIMaster)
- **Judge Model:** `gpt-5.6-luna` (via APIMaster)
- **Embeddings Model:** `nvidia/llama-nemotron-embed-vl-1b-v2` (via NVIDIA)
- **Reranker Model:** `BAAI/bge-reranker-v2-m3` (Local via CrossEncoder)
- **Vector Database:** Chroma (Local Persistence)
- **Evaluation Framework:** DeepEval
- **Tracing & Observability:** LangSmith & LangChain

---

## Offline Evaluation

The offline evaluation suite acts as our automated CI/CD safety net. Before any code or prompt changes are pushed to production, we run the candidate pipeline against curated **Golden Datasets** to verify its performance across various dimensions.

### Core Concepts

**Ground Truth (GT) / Golden Datasets**  
A "Golden Dataset" is a curated set of test cases containing edge-cases, common user queries, and known adversarial attacks. A "Ground Truth" (or Reference) is the ideal, pre-verified expected output (e.g., the exact correct answer or the exact required context chunks) for a given query.

**Reference-Based vs. Reference-Free Metrics**
- **Reference-Based Metrics:** Require Ground Truth to calculate a score. The pipeline's output is directly compared against the human-verified reference to measure correctness or recall.
- **Reference-Free Metrics:** Do not require a pre-defined "perfect" answer. These metrics (like Toxicity or Faithfulness) are calculated solely by analyzing the relationship between the user's query, the retrieved context, and the generated response using an LLM-as-a-judge.

### Evaluation Methodology

Our suite leverages an LLM-as-a-judge approach (using a dedicated `gpt-5.6-luna` proxy model) combined with DeepEval to grade the pipeline on a strict `0.0` to `1.0` scale. **Higher scores are always better** (e.g., a Toxicity score of `1.0` means perfectly safe, whereas `0.0` means highly toxic). Most general metrics require a threshold of `>= 0.7` to pass, while strict safety metrics require `>= 0.95`.

### Offline Metrics Breakdown

Before diving into the metrics, it is important to understand the difference between **Isolated** and **Live** testing:
- **Isolated Evaluation:** Tests a specific component (like the Generator) by feeding it a perfect, hand-curated "Golden Context" instead of the actual retrieved documents. This isolates blame: if the Generator fails here, it is purely a prompting or LLM failure, not a symptom of bad retrieval.
- **Live Evaluation:** Tests the component organically. The Generator is fed the *actual* context returned by the live Retriever. This measures the true end-to-end performance of the integrated pipeline.

We evaluate the pipeline across several distinct matrices (metrics), each using specific criteria calculated by the LLM Judge:

**1. Retriever Layer (`evals/eval_retriever.py`)**
- **Contextual Precision (Reference-Based):** Evaluates if the most relevant ground-truth chunks were ranked at the very top of the retrieved context. It calculates a score by penalizing the system if the correct chunks are buried at the bottom.
- **Contextual Recall (Reference-Based):** Evaluates whether the retrieved context contains *all* the necessary information required to answer the query, compared against the Ground Truth context.

**2. Generator Layer (`evals/eval_generator.py`)**
- **Faithfulness (Reference-Free, Isolation):** Evaluates the generator in isolation using golden context. Extracts factual claims made in the generated answer and cross-references them against the *perfect* ground truth context.
- **Answer Relevancy (Reference-Free, Isolation):** Evaluates if the generator directly addresses the prompt using golden context, penalizing evasive answers.

**3. RAG Pipeline / Triad (`evals/eval_rag_pipeline.py`)**
- **Contextual Relevancy (Reference-Free):** Evaluates whether the retrieved context is genuinely relevant to the user's query, ensuring the vector search isn't pulling in useless noise.
- **Faithfulness (Reference-Free, Live):** The judge extracts factual claims from the final generated answer and ensures they are strictly supported by the *live retrieved context* (not ground truth), penalizing hallucinations.
- **Answer Relevancy (Reference-Free, Live):** Evaluates if the final live answer directly and completely addresses the user's original query.

**4. Application Layer (`evals/eval_application.py`)**
- **GEval Correctness (Reference-Based):** Uses a custom prompt rubric to ask the LLM judge to compare the generated answer directly against the Ground Truth answer for strict factual alignment.
- **Completeness (Reference-Based):** Evaluates if the generated answer comprehensively covers all nuances and multi-part questions present in the Ground Truth without leaving out details.

**5. Safety & Security (`evals/eval_safety.py`)**
- **PII Leakage (Reference-Free):** The judge scans the output for sensitive Personally Identifiable Information (like phone numbers or addresses) and fails the run if the system leaked any private data instead of declining.
- **Domain Scope (Reference-Free):** Evaluates if the user's query is outside the system's intended domain (e.g., asking a history bot about modern coding) and ensures the pipeline politely refuses to answer.
- **Toxicity (Reference-Free):** Scans the answer for hate speech, bias, or harmful language.

**6. Operational Layer (`evals/eval_ops.py`)**
- **Reliability:** Tracks the success rate and error rate of the pipeline, ensuring system stability (e.g., failing if API timeouts or crashes occur).
- **Latency:** Measures the real-time End-to-End processing duration and fails the test if it breaches our target budget (e.g., `p95 > 15 seconds`).
- **Cost:** Uses LangSmith tracing to estimate token usage and ensures the API cost per query remains under the defined financial budget limits.

### Running the Suite

- **Run the evaluation suite:**  
  `python -m evals.run_suite`  
  *This master script executes all the evaluation files above, aggregates the scores, and saves them to `baselines/candidate.json`.*

- **Compare candidate vs baseline:**  
  `python -m evals.compare`  
  *This script mathematically compares the newly generated candidate results against the locked `baseline.json`. It flags regressions (declining scores) and improvements, using custom variance tolerances to prevent false-negative build failures caused by LLM judge non-determinism.*

![Offline Comparison Screenshot](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.18.22%E2%80%AFAM.png?raw=true)

---

## Online Evaluation & LangSmith Tracing

In production, we use LangSmith to monitor the pipeline across **6 critical metrics**. These metrics are visualized via dashboards and tied to alerting systems.

### 1. Native LangSmith Auto-Evaluators
The following operational metrics are tracked natively using LangSmith's built-in evaluators:
- **Cost**
- **Latency**
- **Toxicity**

### 2. The Custom RAG Triad Worker
The remaining evaluation metrics (The RAG Triad) require our custom `gpt-5.6-luna` proxy judge. We run a background daemon that periodically fetches recent traces, grades them, and pushes the scores back to LangSmith:
- **Faithfulness**
- **Answer Relevancy**
- **Contextual Relevancy**

**To start the online worker:**
```bash
python -m evals.eval_online
```
*(Note: For production, this script can be scheduled as a Cron job).*

- **Average Faithfulness score chart:**  
![Average Faithfulness score chart](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.15.50%E2%80%AFAM.png?raw=true)

- **Average Contextual Relevancy score chart:**  
![Average Contextual Relevancy score chart](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.16.03%E2%80%AFAM.png?raw=true)

- **Average Answer Relevancy score chart:**  
![Average Answer Relevancy score chart](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.16.14%E2%80%AFAM.png?raw=true)

- **Average Toxicity score chart:**  
![Average Toxicity score chart](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.16.24%E2%80%AFAM.png?raw=true)

- **Average Cost chart:**  
![Average Cost chart](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.16.36%E2%80%AFAM.png?raw=true)

- **p95 score chart:**  
![p95 score chart](https://github.com/yashh2417/rag-evaluation-and-tracing-for-optimization/blob/main/llm-evals-ss/Screenshot%202026-09-28%20at%2010.16.48%E2%80%AFAM.png?raw=true)

---

## Alerts & Continuous Improvement

To ensure the system is constantly improving, we have implemented a tight feedback loop:
- **Threshold Alerts:** We have configured LangSmith alerts to trigger if Latency/Toxicity breach limits, or if Triad scores fall below the 0.7 threshold.
- **Expanding Datasets:** Any traces that crash, fail safety gates, or receive poor RAG triad scores are automatically routed to our LangSmith Datasets. These edge-cases are periodically pulled down to expand our offline evaluation datasets, ensuring the pipeline learns from production failures.
