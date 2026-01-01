"""
3_ball3_ml/predict.py - ball3 예측

ball1, ball2, ball6가 주어졌을 때 ball3 예측
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data


def predict_ball3(
    used_balls: Set[int],
    past_data: List[Dict],
    top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    ball3 예측

    Args:
        used_balls: 이미 사용된 번호들 {ball1, ball2, ball6}
        past_data: 과거 데이터
        top_k: 상위 K개 반환

    Returns:
        [(ball3, score), ...] 점수 높은 순
    """
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    recent_20 = past_data[-20:] if len(past_data) >= 20 else past_data

    # ball3 빈도
    freq_10 = Counter(d['ball3'] for d in recent_10)
    freq_20 = Counter(d['ball3'] for d in recent_20)

    candidates = []
    for num in range(1, 46):
        if num in used_balls:
            continue

        # 빈도 점수
        freq_score = freq_10.get(num, 0) * 2 + freq_20.get(num, 0)
        candidates.append((num, freq_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]
