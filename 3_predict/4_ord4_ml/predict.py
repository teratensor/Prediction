"""
4_ord4_ml/predict.py - ord4 예측 (ML 기반, 가장 어려운 위치)

전략:
- LightGBM 모델 기반 예측
- 빈도 기반 보정
- 범위: ord1 + 3 <= ord4 <= ord6 - 2
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data

MODEL_PATH = Path(__file__).parent.parent / "models" / "ord4_model.pkl"
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


def predict_ord4(
    ord1: int,
    ord6: int,
    past_data: List[Dict],
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """
    ord4 예측 (ML + 휴리스틱 앙상블)

    Args:
        ord1: 첫 번째 번호
        ord6: 여섯 번째 번호
        past_data: 과거 데이터
        top_k: 상위 K개 반환

    Returns:
        [(ord4, score), ...] 점수 높은 순
    """
    span = ord6 - ord1

    # 유효 범위: ord1+3 <= ord4 <= ord6-2
    min_ord4 = ord1 + 3
    max_ord4 = ord6 - 2

    if min_ord4 > max_ord4:
        return []

    # 최근 데이터
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    freq_ord4 = Counter(d['ord4'] for d in recent_10)

    # 비율 통계
    ratios = []
    for d in past_data[-50:]:
        d_span = d['ord6'] - d['ord1']
        if d_span > 0:
            ratio = (d['ord4'] - d['ord1']) / d_span
            ratios.append(ratio)
    avg_ratio = np.mean(ratios) if ratios else 0.6
    expected = ord1 + span * avg_ratio

    # 이전 회차
    prev = past_data[-1] if past_data else None
    prev_value = prev['ord4'] if prev else 0

    model = get_model()

    candidates = []
    for ord4 in range(min_ord4, max_ord4 + 1):
        if model is not None:
            features = [
                ord1,
                ord6,
                span,
                avg_ratio,
                expected,
                freq_ord4.get(ord4, 0),
                0,  # freq_20 placeholder
                min_ord4,
                max_ord4,
                max_ord4 - min_ord4,
                prev_value,
                ord4 - prev_value if prev else 0,
                1 if prev and ord4 in [prev[f'ord{i}'] for i in range(1, 7)] else 0,
                ord4 % 10,
                1 if ord4 in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43} else 0,
                ord4 % 2,
                0, 0, 0  # known_ords placeholder
            ]
            pred = model.predict([features])[0]
            ml_score = 1.0 / (1.0 + abs(ord4 - pred))
        else:
            ml_score = 1.0 / (1.0 + abs(ord4 - expected))

        # 빈도 보너스
        freq_score = freq_ord4.get(ord4, 0) * 0.1

        total_score = ml_score + freq_score
        candidates.append((ord4, total_score))

    # 점수 높은 순 정렬
    candidates.sort(key=lambda x: -x[1])

    return candidates[:top_k]


def get_ord4_ratio_stats(past_data: List[Dict]) -> Dict:
    """ord4 위치 비율 통계"""
    ratios = []
    for d in past_data:
        span = d['ord6'] - d['ord1']
        if span > 0:
            ratio = (d['ord4'] - d['ord1']) / span
            ratios.append(ratio)

    if not ratios:
        return {'mean': 0.6, 'min': 0.4, 'max': 0.8}

    return {
        'mean': sum(ratios) / len(ratios),
        'min': min(ratios),
        'max': max(ratios),
        'count': len(ratios)
    }


if __name__ == '__main__':
    data = load_winning_data()
    target = 1200

    # 통계 확인
    past = [d for d in data if d['round'] < target]
    stats = get_ord4_ratio_stats(past)
    print(f"=== ord4 위치 통계 ===")
    print(f"평균 비율: {stats['mean']:.3f}")
    print(f"범위: {stats['min']:.3f} ~ {stats['max']:.3f}")

    # 예측 테스트
    actual = next((d for d in data if d['round'] == target), None)
    if actual:
        ord1, ord6 = actual['ord1'], actual['ord6']
        print(f"\n=== {target}회차 ord4 예측 ===")
        print(f"ord1={ord1}, ord6={ord6}, span={ord6-ord1}")

        predictions = predict_ord4(ord1, ord6, past, top_k=10)
        print("\nTop-10 예측:")
        for i, (o4, score) in enumerate(predictions, 1):
            marker = " ← 정답" if o4 == actual['ord4'] else ""
            print(f"  {i:2d}. ord4={o4:2d} score={score:.3f}{marker}")

        print(f"\n실제 ord4: {actual['ord4']}")
