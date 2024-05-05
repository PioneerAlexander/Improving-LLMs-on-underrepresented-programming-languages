"""
    This module contains the metrics functions, which will be used for
    analysing the performance. These metrics are BLEU score, Exact Match,
    and Edit Similarity.
"""
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import evaluate
import Levenshtein


def bleu_score(reference: str, pred: str) -> float:
    """
        Calculates BLEU Score https://en.wikipedia.org/wiki/BLEU
    """
    return sentence_bleu(
        [reference.split()],
        pred.split()
    )


def exact_match(reference: str, pred: str) -> float:
    """
        Calculates the exact match score using function from HuggingFace
        https://huggingface.co/spaces/evaluate-metric/exact_match
    """
    return evaluate.load("exact_match").compute(
        references=[reference],
        predictions=[pred]
    )


def edit_similarity(reference: str, pred: str) -> float:
    """
        Calculates the distance between reference and prediction texts
    """
    distance = Levenshtein.distance(reference, pred)

    # Normalizing the distance to get a similarity score
    max_len = max(len(reference), len(pred))

    if max_len == 0:
        return 1.0  # both strings are empty

    similarity = 1 - (distance / max_len)
    return similarity
