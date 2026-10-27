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

df["response"] = df["question"].apply(lambda q: chat(q, MODEL, max_tokens=300))

print(df)
df.to_csv("responses.csv", index=False)
