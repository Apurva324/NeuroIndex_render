from vectordb import DocumentDB
from NeuroIndex.client import OllamaClient
import config


def main():
    ollama = OllamaClient()
    db = DocumentDB(dim=config.DIMS)

    documents = db.list_documents()

    print(f"Found {len(documents)} documents")
    print(f"Target dimension: {config.DIMS}")
    print()

    if not ollama.is_available():
        print("ERROR: Ollama is not available.")
        print("Run: ollama serve")
        return

    success = 0
    failed = 0

    for document in documents:
        document_id = document["id"]
        title = document.get("title", "")
        text = document.get("text", "")

        print(f"Embedding {document_id}: {title}")

        embedding = ollama.embed(text)

        if not embedding:
            print("  FAILED: Ollama returned an empty embedding")
            failed += 1
            continue

        if len(embedding) != config.DIMS:
            print(
                f"  FAILED: expected {config.DIMS}, "
                f"got {len(embedding)}"
            )
            failed += 1
            continue

        db.insert(
            document_id=document_id,
            title=title,
            text=text,
            embedding=embedding,
            metadata=document.get("metadata", {}),
        )

        success += 1
        print(f"  OK: {len(embedding)} dimensions")

    print()
    print("=" * 50)
    print(f"Successful: {success}")
    print(f"Failed:     {failed}")
    print(f"Total:      {len(documents)}")
    print("=" * 50)


if __name__ == "__main__":
    main()