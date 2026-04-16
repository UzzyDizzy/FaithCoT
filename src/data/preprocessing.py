#src/data/preprocessing.py
"""
Preprocessing.

Converts raw datasets into unified format:
{
    "id": str,
    "question": str,
    "answer": str
}
"""

import re
from configs.benchmark_config import BENCHMARK_REGISTRY, PROMPT_TEMPLATES

def preprocess_dataset(dataset, bench_key, prompt_format="zero_shot_cot"):
    config = BENCHMARK_REGISTRY[bench_key]
    template = PROMPT_TEMPLATES[prompt_format][bench_key]

    processed = []

    for i, d in enumerate(dataset):
        question = d.get(config.question_field, "")

        # ---- HANDLE SPECIAL CASES ----
        if bench_key == "arc_challenge":
            choices = d.get("choices", {})
            if isinstance(choices, dict):
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                options = "\n".join(
                    [f"({l}) {t}" for l, t in zip(labels, texts)]
                )
            else:
                options = ""

            prompt = template.format(question=question, choices=options)

        elif bench_key == "folio":
            conclusion = d.get("conclusion", "")
            prompt = template.format(question=question, conclusion=conclusion)

        else:
            prompt = template.format(question=question)

        answer = d.get(config.answer_field)

        # 🔥 HARD FIX
        if answer is None:
            print(f"[PREPROCESS DEBUG] Missing answer at index {i}")
            print("RAW SAMPLE:", d)
            answer = ""

        # -------------------------
        # NORMALIZATION
        # -------------------------

        if bench_key == "gsm8k":
            if isinstance(answer, str) and "####" in answer:
                answer = answer.split("####")[-1].strip()

        elif bench_key == "strategyqa":
            val = d.get(config.answer_field)

            if isinstance(val, bool):
                answer = "yes" if val else "no"
            else:
                val = str(val).lower()
                if val in ["true", "yes"]:
                    answer = "yes"
                elif val in ["false", "no"]:
                    answer = "no"
                else:
                    answer = val

        elif bench_key == "math":
            boxed = re.findall(r"\\boxed\{([^}]+)\}", str(answer))
            if boxed:
                answer = boxed[-1]

        #  FINAL SAFETY
        if answer is None or str(answer).strip() == "":
            print(f"[PREPROCESS WARNING] Empty answer at index {i}")
            answer = "0"   # prevent SIG crash

        processed.append({
            "id": f"{bench_key}_{i}",
            "question": question,
            "prompt": prompt,
            "gold_answer": str(answer),
            "answer_type": config.answer_type,
        })

    return processed

# Optional loader for cached processed data
def load_processed_data(path):
    import json

    with open(path, "r") as f:
        return json.load(f)