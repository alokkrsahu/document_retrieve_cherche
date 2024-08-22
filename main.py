# main.py

from .dpr import DPRRetriever
from .encoder import DocumentRetriever as EncoderDocumentRetriever
from .golden import DocumentRetriever as GoldenDocumentRetriever

def run_dpr_retriever(documents, query, k, device="cpu"):
    print("DPRRetriever results:")
    dpr_retriever = DPRRetriever(documents, device=device)
    results = dpr_retriever.retrieve(query, k=k)
    print(results)
    return results

def run_encoder_retriever(documents, query, k, model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"):
    print("\nEncoderDocumentRetriever results:")
    encoder_retriever = EncoderDocumentRetriever(documents, model_name=model_name, device=device)
    results = encoder_retriever.retrieve(query, k=k)
    print(results)
    return results

def run_golden_retriever(documents, query, method, k, model_name="sentence-transformers/all-mpnet-base-v2", use_gpu=False):
    print(f"\nGoldenDocumentRetriever results using method '{method}':")
    try:
        golden_retriever = GoldenDocumentRetriever(
            method=method,
            documents=documents,
            on=["text"],
            use_gpu=use_gpu,
            model_name=model_name
        )
        results = golden_retriever.retrieve(query, k=k)
        print(results)
        return results
    except Exception as e:
        print("ERROR")
        print(e)

def main(documents, query, method, k):
    if method == "dpr":
        run_dpr_retriever(documents, query, k)
    elif method == "encoder":
        run_encoder_retriever(documents, query, k)
    elif method in ["bm25", "tfidf", "flash", "lunr", "fuzz", "embedding"]:
        run_golden_retriever(documents, query, method, k)
    else:
        print("Invalid method specified.")

if __name__ == "__main__":
    # Example parameters for standalone execution
    #documents = [
    #    {"id": 0, "text": "Paris is the capital and most populous city of France", "title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"},
    #    {"id": 1, "text": "Paris has been one of Europe's major centres of finance, diplomacy, commerce, fashion, gastronomy, science, and arts.", "title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"},
    #    {"id": 2, "text": "The City of Paris is the centre and seat of government of the region and province of Île-de-France.", "title": "Paris", "url": "https://en.wikipedia.org/wiki/Paris"}
    #]
    #query = "Paris"
    #method = "bm25"  # Replace with "dpr", "encoder", or any valid Golden Retriever method
    #k = 3  # Number of results to retrieve

    # Call main with parameters
    main(documents, query, method, k)
