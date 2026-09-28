from reranker import RerankingRetriever
from generator import generate, compress_context
from langsmith import traceable



class RagPipeline:
    def __init__(self, fetch_k=8, top_k=3):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: format docs with metadata for the generator
        raw_context = [
            f"[{doc.metadata.get('source_type', 'unknown').upper()}: {doc.metadata.get('source', 'unknown')}] {doc.page_content}"
            for doc in docs
        ]

        # 3. COMPRESS: scrub out all irrelevant sentences
        # compressed_text = compress_context(query, raw_context)
        # context = [compressed_text] # Package as a list so the generator code doesn't break
        context = raw_context

        # 4. GENERATE: grounded answer from the compressed context
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# quick manual smoke test: python src/rag_pipeline.py
if __name__ == "__main__":
    
    rag = RagPipeline()
    result = rag.invoke("who is egg?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")