import numpy as np

def wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis strings.

    Args:
        reference (str): Ground truth sentence
        hypothesis (str): Predicted sentence

    Returns:
        float: Word Error Rate
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    n = len(ref_words)

    # Initialize the distance matrix
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # Deletion
                d[i][j - 1] + 1,      # Insertion
                d[i - 1][j - 1] + cost  # Substitution
            )

    return d[len(ref_words)][len(hyp_words)] / max(n, 1)  # Avoid division by zero
