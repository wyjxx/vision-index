"""Retrieval metrics using the original result order and held-out pair labels."""

from math import log2


def _validate_inputs(predicted_ids: list[str], relevant_products: dict[str, int], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if any(not isinstance(pid, str) or not pid.strip() for pid in predicted_ids):
        raise ValueError("Predicted product IDs must be nonempty strings")
    if len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError("Predicted product IDs must be unique")
    for product_id, position in relevant_products.items():
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("Relevant product IDs must be nonempty strings")
        if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= 100:
            raise ValueError("Reference position must be an integer from 1 to 100")


def recall_at_k(predicted_ids: list[str], relevant_products: dict[str, int], k: int) -> float:
    """
    predicted_ids = X, B, A, C
    all_relevant_products =  X, B, C
    test_relevant_products = X, C
    k = 3 
    recall@k = 1/2
    """
    _validate_inputs(predicted_ids, relevant_products, k)
    if not relevant_products:
        return 0.0
    hits = sum(product_id in relevant_products for product_id in predicted_ids[:k])
    return hits / len(relevant_products)


def ndcg_at_k(predicted_ids: list[str], relevant_products: dict[str, int], k: int) -> float:
    """Measure if model rank more relevant product first

    predicted_ids = ["X", "B", "A", "C"]
    test_relevant_products = {"X": 1, "C": 51}
    k = 3
    gain = 101 - position
    X gain = 100
    C gain = 50

    DCG@K = Σ gain / log₂(model_rank + 1)
    DCG@3 = 100/log2(2) + 0/log2(3) + 0/log2(4) = 100
    IDCG@3 = 100/log2(2) + 50/log2(3) ≈ 131.5465
    nDCG@3 = 100 / 131.5465 ≈ 0.7602
    """
    _validate_inputs(predicted_ids, relevant_products, k)
    if not relevant_products:
        return 0.0
    gains = {product_id: 101 - position for product_id, position in relevant_products.items()}
    dcg = sum(
        gains.get(product_id, 0) / log2(rank + 1)
        for rank, product_id in enumerate(predicted_ids[:k], start=1)
    )
    ideal_gains = sorted(gains.values(), reverse=True)[:k]
    ideal_dcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(ideal_gains, start=1))
    return dcg / ideal_dcg
