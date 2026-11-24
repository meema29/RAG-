"""Terminal chatbot. Usage: python chatbot.py"""
import config
from rag import ask


def main() -> None:
    print("Pharmaly chatbot - ask about Pharmaly products. Type 'exit' to quit.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            result = ask(question, history)
        except Exception as exc:  # API / index errors: report, keep running
            print(f"\n[error] {exc}")
            continue

        print(f"\nPharmaly: {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  - {s['name']} (similarity {s['similarity']:.2f})\n    {s['url']}")
        else:
            best = result["max_similarity"]
            if best is not None:
                print(f"\n(best similarity {best:.2f}, threshold {config.SIMILARITY_THRESHOLD})")
        history.append((question, result["answer"]))
        history = history[-config.MAX_HISTORY_TURNS:]
    print("Goodbye!")


if __name__ == "__main__":
    main()
