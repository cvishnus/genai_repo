import time

import pandas as pd

from claude_client import chat

MODEL = "claude-haiku-5-5"

df = pd.DataFrame({"question": [
    "What is overfitting?",
    "What is a p-value?",
    "Explain precision vs recall.",
    "What is cross-validation?",
    "What is regularization?",
]})



def timed_chat(question: str) -> tuple[str, float]:
    start = time.perf_counter()
    answer = chat(question, MODEL, max_tokens=300)
    return answer, round(time.perf_counter() - start, 2)


df[["response", "latency_s"]] = df["question"].apply(lambda q: pd.Series(timed_chat(q)))

print(df)
df.to_csv("responses.csv", index=False)
