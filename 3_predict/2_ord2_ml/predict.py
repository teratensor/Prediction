"""
2_ord2_ml/predict.py - ord2 예측 (ML 기반)

범위: ord1 < ord2 < ord3 < ord4
     → ord1 + 1 <= ord2 <= ord4 - 2
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data, get_cluster_weights

MODEL_PATH = Path(__file__).parent.parent / "models" / "ord2_model.pkl"
_MODEL = None

def _load_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def predict_ord2(
    ord1: int,
    ord4: int,
    past_data: List[Dict],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    ord2 예측 (ML + 휴리스틱 앙상블)

    Args:
        ord1: 첫 번째 번호
        ord4: 네 번째 번호
        past_data: 과거 데이터
        top_k: 상위 K개 반환

    Returns:
        [(ord2, score), ...] 점수 높은 순
    """
    # 유효 범위: ord1+1 <= ord2 <= ord4-2
    min_ord2 = ord1 + 1
    max_ord2 = ord4 - 2

    if min_ord2 > max_ord2:
        return []

    # 최근 데이터
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    freq_ord2 = Counter(d['ord2'] for d in recent_10)

    # 비율 통계
    ord6 = ord4 + 5  # 추정
    span = ord6 - ord1
    ratios = []
    for d in past_data[-50:]:
        d_span = d['ord6'] - d['ord1']
        if d_span > 0:
            ratio = (d['ord2'] - d['ord1']) / d_span
            ratios.append(ratio)
    avg_ratio = np.mean(ratios) if ratios else 0.3
    expected = ord1 + span * avg_ratio

    # 이전 회차
    prev = past_data[-1] if past_data else None
    prev_value = prev['ord2'] if prev else 0

    model = get_model()

    candidates = []
    for ord2 in range(min_ord2, max_ord2 + 1):
        if model is not None:
            features = [
                ord1,
                ord6,
                span,
                avg_ratio,
                expected,
                freq_ord2.get(ord2, 0),
                0,  # freq_20 placeholder
                min_ord2,
                max_ord2,
                max_ord2 - min_ord2,
                prev_value,
                ord2 - prev_value if prev else 0,
                1 if prev and ord2 in [prev[f'ord{i}'] for i in range(1, 7)] else 0,
                ord2 % 10,
                1 if ord2 in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43} else 0,
                ord2 % 2,
                0, 0, 0  # known_ords placeholder
            ]
            pred = model.predict([features])[0]
            ml_score = 1.0 / (1.0 + abs(ord2 - pred))
        else:
            ml_score = 1.0 / (1.0 + abs(ord2 - expected))

        # 빈도 보너스
        freq_score = freq_ord2.get(ord2, 0) * 0.1

        # 구간 선호도 (10-19 구간이 48.3%로 최빈)
        range_score = 0.15 if 10 <= ord2 <= 19 else 0.0

        total_score = ml_score + freq_score + range_score
        candidates.append((ord2, total_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


if __name__ == '__main__':
    data = load_winning_data()
    target = 1200

    actual = next((d for d in data if d['round'] == target), None)
    past = [d for d in data if d['round'] < target]

    if actual:
        ord1, ord4 = actual['ord1'], actual['ord4']
        print(f"=== {target}회차 ord2 예측 ===")
        print(f"ord1={ord1}, ord4={ord4}")

        predictions = predict_ord2(ord1, ord4, past, top_k=10)
        print("\nTop-10 예측:")
        for i, (o2, score) in enumerate(predictions, 1):
            marker = " ← 정답" if o2 == actual['ord2'] else ""
            print(f"  {i:2d}. ord2={o2:2d} score={score:.3f}{marker}")

        print(f"\n실제 ord2: {actual['ord2']}")
