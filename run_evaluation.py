from src.eval.runner import evaluate_texts

if __name__ == "__main__":
    texts = [
        (
            "Artificial intelligence is transforming medicine. "
            "Machine learning models help doctors detect diseases early. "
            "Early diagnosis improves treatment outcomes. "
            "Today it is raining and traffic is heavy."
        ),
        (
            "Renewable energy sources like solar and wind reduce carbon emissions. "
            "Energy storage is important for balancing supply and demand. "
            "Grid modernization improves reliability. "
            "Some people also enjoy hiking on weekends."
        ),
    ]

    results = evaluate_texts(texts, k_extractive=2, k_final=4, llm_target_sentences=3)

    for text_key, items in results.items():
        print("\n" + "=" * 80)
        print(text_key)
        print("=" * 80)
        for r in items:
            print(f"\n--- {r.name.upper()} ---")
            print(f"runtime_sec: {r.runtime_sec:.4f}")
            print(f"words: {r.words}")
            print(f"redundancy: {r.redundancy:.4f}")
            print(f"coverage: {r.coverage:.4f}")
            print("summary:", r.summary)
