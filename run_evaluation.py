from __future__ import annotations

import csv
from pathlib import Path

from src.eval.runner import evaluate_texts


def main() -> None:
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

    results = evaluate_texts(
        texts,
        k_extractive=2,
        k_final=4,
        llm_target_sentences=3,
    )


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

    #  SAVE CSV 
    output_path = Path("evaluation_results.csv")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "text_id",
                "method",
                "runtime_sec",
                "words",
                "redundancy",
                "coverage",
                "summary",
            ]
        )

        for text_key, items in results.items():
            for r in items:
                writer.writerow(
                    [
                        text_key,
                        r.name,
                        f"{r.runtime_sec:.6f}",
                        r.words,
                        f"{r.redundancy:.6f}",
                        f"{r.coverage:.6f}",
                        r.summary,
                    ]
                )

    print("\nSaved evaluation results to:", output_path.resolve())


if __name__ == "__main__":
    main()
