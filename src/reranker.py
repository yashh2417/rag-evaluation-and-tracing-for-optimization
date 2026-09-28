from sentence_transformers import CrossEncoder
from retriever import load_store
from langsmith import traceable

# small, fast, CPU-friendly reranker. Downloads once (~80MB) on first run.
CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


class RerankingRetriever:
    def __init__(self, fetch_k=8, top_k=3):
        self.store = load_store()
        self.reranker = CrossEncoder(CROSS_ENCODER)
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK: score each (query, chunk) pair TOGETHER with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]

if __name__ == "__main__":
    print("Initializing RerankingRetriever...")
    retriever = RerankingRetriever(fetch_k=12, top_k=5)
    
    query = "who was the first king of westeros?"
    print(f"\nQuerying: '{query}'")
    print("-" * 50)
    
    results = retriever.invoke(query)
    print(len(results))
    
    print("--- Top Results after Reranking ---")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.metadata.get('source_type', 'unknown').upper()}: {r.metadata.get('source', 'unknown')}]")
        print(f"   {r.page_content[:150]}...\n")
        