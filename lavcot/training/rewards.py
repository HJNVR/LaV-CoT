from __future__ import annotations

import re
from typing import Any, List, Sequence


def completion_texts(completions: Sequence[Any]) -> List[str]:
    texts: List[str] = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        elif isinstance(completion, list) and completion:
            content = completion[0].get("content", "")
            if isinstance(content, list):
                content = "".join(x.get("text", "") for x in content)
            texts.append(str(content))
        else:
            texts.append(str(completion))
    return texts


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.I)
    return matches[-1].strip() if matches else ""


def format_reward(completions: Sequence[Any], **_: Any) -> List[float]:
    pattern = re.compile(
        r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL | re.I
    )
    return [1.0 if pattern.match(text) else 0.0 for text in completion_texts(completions)]


def language_reward(
    completions: Sequence[Any], lang: Sequence[str], **_: Any
) -> List[float]:
    """Reward an explicit language marker; language-ID packages remain optional."""
    return [
        1.0 if target.casefold() in text.casefold() else 0.0
        for text, target in zip(completion_texts(completions), lang)
    ]


def answer_reward(
    completions: Sequence[Any], answer: Sequence[str], **_: Any
) -> List[float]:
    scores: List[float] = []
    for text, target in zip(completion_texts(completions), answer):
        prediction = " ".join(extract_answer(text).casefold().split())
        gold = " ".join(str(target).casefold().split())
        scores.append(float(bool(gold) and prediction == gold))
    return scores


def count_reward(
    completions: Sequence[Any], gt_counts: Sequence[int], **_: Any
) -> List[float]:
    scores: List[float] = []
    for text, gold in zip(completion_texts(completions), gt_counts):
        match = re.search(r"\\obj\{(\d+)\}", text)
        prediction = int(match.group(1)) if match else 0
        gold = int(gold)
        scores.append(max(0.0, 1.0 - abs(prediction - gold) / max(gold, 1)))
    return scores
