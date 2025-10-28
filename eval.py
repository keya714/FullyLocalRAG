#!/usr/bin/env python3
"""
Very small eval harness:
- Loads a JSONL with fields: question, must_include (list of strings), nice_to_have (list)
- Runs the RAG and scores inclusion
Usage: python eval.py --file testsamples.jsonl
"""
import json
import subprocess
import argparse
from statistics import mean

def run_query(q: str) -> str:
    # Call our CLI and capture stdout
    res = subprocess.run(
        ["python", "main.py", "--q", q],
        capture_output=True, text=True
    )
    return res.stdout

def score_answer(ans: str, must: list[str], nice: list[str]) -> tuple[float,float]:
    must_hits = [1.0 if s.lower() in ans.lower() else 0.0 for s in must]
    nice_hits = [1.0 if s.lower() in ans.lower() else 0.0 for s in nice]
    return (mean(must_hits) if must_hits else 1.0, mean(nice_hits) if nice_hits else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    must_scores, nice_scores = [], []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["question"]
            ans = run_query(q)
            m, n = score_answer(ans, ex.get("must_include", []), ex.get("nice_to_have", []))
            print(f"Q: {q}\nMust={m:.2f} Nice={n:.2f}\n")
            must_scores.append(m); nice_scores.append(n)
    print(f"Aggregate: Must={mean(must_scores):.2f} Nice={mean(nice_scores):.2f}")

if __name__ == "__main__":
    main()
