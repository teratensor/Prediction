"""
5_ord5_ml/predict.py - ord5 예측 (ML 기반, 가장 쉬운 위치)

범위: ord4 < ord5 < ord6
     → ord4 + 1 <= ord5 <= ord6 - 1

특징: 30-39 구간이 54.6%로 최빈
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data

MODEL_PATH = Path(__file__).parent.parent / "models" / "ord5_model.pkl"
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


def predict_ord5(
    ord4: int,
    ord6: int,
    past_data: List[Dict],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    ord5 예측 (ML + 휴리스틱 앙상블)

    Args:
        ord4: 네 번째 번호
        ord6: 여섯 번째 번호
        past_data: 과거 데이터
        top_k: 상위 K개 반환

    Returns:
        [(ord5, score), ...] 점수 높은 순
    """
    # 유효 범위: ord4+1 <= ord5 <= ord6-1
    min_ord5 = ord4 + 1
    max_ord5 = ord6 - 1

    if min_ord5 > max_ord5:
        return []

    # 최근 데이터
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    freq_ord5 = Counter(d['ord5'] for d in recent_10)

    # 비율 통계
    ord1 = ord4 - 10  # 추정
    span = ord6 - ord1
    ratios = []
    for d in past_data[-50:]:
        d_span = d['ord6'] - d['ord1']
        if d_span > 0:
            ratio = (d['ord5'] - d['ord1']) / d_span
            ratios.append(ratio)
    avg_ratio = np.mean(ratios) if ratios else 0.8
    expected = ord1 + span * avg_ratio

    # 이전 회차
    prev = past_data[-1] if past_data else None
    prev_value = prev['ord5'] if prev else 0

    model = get_model()

    candidates = []
    for ord5 in range(min_ord5, max_ord5 + 1):
        if model is not None:
            features = [
                ord1,
                ord6,
                span,
                avg_ratio,
                expected,
                freq_ord5.get(ord5, 0),
                0,  # freq_20 placeholder
                min_ord5,
                max_ord5,
                max_ord5 - min_ord5,
                prev_value,
                ord5 - prev_value if prev else 0,
                1 if prev and ord5 in [prev[f'ord{i}'] for i in range(1, 7)] else 0,
                ord5 % 10,
                1 if ord5 in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43} else 0,
                ord5 % 2,
                0, 0, ord4  # known_ords: ord4만 알려짐
            ]
            pred = model.predict([features])[0]
            ml_score = 1.0 / (1.0 + abs(ord5 - pred))
        else:
            ml_score = 1.0 / (1.0 + abs(ord5 - expected))

        # 빈도 보너스
        freq_score = freq_ord5.get(ord5, 0) * 0.1

        # 구간 선호도 (30-39 구간이 54.6%로 최빈)
        if 30 <= ord5 <= 39:
            range_score = 0.2
        elif 28 <= ord5 <= 36:
            range_score = 0.1
        else:
            range_score = 0.0

        total_score = ml_score + freq_score + range_score
        candidates.append((ord5, total_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


if __name__ == '__main__':
    data = load_winning_data()
    target = 1200

    actual = next((d for d in data if d['round'] == target), None)
    past = [d for d in data if d['round'] < target]

    if actual:
        ord4, ord6 = actual['ord4'], actual['ord6']
        print(f"=== {target}회차 ord5 예측 ===")
        print(f"ord4={ord4}, ord6={ord6}")

        predictions = predict_ord5(ord4, ord6, past, top_k=10)
        print("\nTop-10 예측:")
        for i, (o5, score) in enumerate(predictions, 1):
            marker = " ← 정답" if o5 == actual['ord5'] else ""
            print(f"  {i:2d}. ord5={o5:2d} score={score:.3f}{marker}")

        print(f"\n실제 ord5: {actual['ord5']}")
