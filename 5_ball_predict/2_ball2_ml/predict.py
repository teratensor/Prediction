"""
2_ball2_ml/predict.py - ball2 예측

ball1, ball6가 주어졌을 때 ball2 예측
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data


def predict_ball2(
    ball1: int,
    ball6: int,
    past_data: List[Dict],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    ball2 예측 (ball1, ball6 조건부)

    Args:
        ball1: 첫 번째 공 번호
        ball6: 여섯 번째 공 번호
        past_data: 과거 데이터
        top_k: 상위 K개 반환

    Returns:
        [(ball2, score), ...] 점수 높은 순
    """
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    recent_20 = past_data[-20:] if len(past_data) >= 20 else past_data

    # ball2 빈도
    freq_10 = Counter(d['ball2'] for d in recent_10)
    freq_20 = Counter(d['ball2'] for d in recent_20)

    # 사용된 번호 제외
    used = {ball1, ball6}

    candidates = []
    for num in range(1, 46):
        if num in used:
            continue

        # 빈도 점수
        freq_score = freq_10.get(num, 0) * 2 + freq_20.get(num, 0)
        candidates.append((num, freq_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]
