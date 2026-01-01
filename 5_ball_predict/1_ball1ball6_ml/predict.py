"""
1_ball1ball6_ml/predict.py - (ball1, ball6) 쌍 예측

Ball은 추첨 순서 (1~6번째로 나온 공)
Ord와 달리 정렬되지 않으며 1-45 범위의 모든 숫자 가능
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# 상위 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data


def predict_ball_individual(
    past_data: List[Dict],
    ball_pos: str,  # 'ball1', 'ball2', ..., 'ball6'
    top_k: int = 20
) -> List[Tuple[int, float]]:
    """
    개별 ball 위치 예측 (빈도 기반)

    Args:
        past_data: 과거 데이터
        ball_pos: 예측할 ball 위치 ('ball1' ~ 'ball6')
        top_k: 상위 K개 반환

    Returns:
        [(ball_number, score), ...] 점수 높은 순
    """
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    recent_20 = past_data[-20:] if len(past_data) >= 20 else past_data

    freq_10 = Counter(d[ball_pos] for d in recent_10)
    freq_20 = Counter(d[ball_pos] for d in recent_20)

    candidates = []
    for num in range(1, 46):
        # 빈도 점수 (최근 10회 가중치 2배)
        freq_score = freq_10.get(num, 0) * 2 + freq_20.get(num, 0)
        candidates.append((num, freq_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def predict_ball1ball6_v2(
    target_round: int,
    all_data: List[Dict],
    top_ball1: int = 20,
    top_ball6: int = 20
) -> List[Tuple[int, int, float]]:
    """
    (ball1, ball6) 쌍 예측 - 개별 예측 후 조합

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        top_ball1: ball1 상위 K개
        top_ball6: ball6 상위 K개

    Returns:
        [(ball1, ball6, score), ...]
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    # 개별 예측
    ball1_candidates = predict_ball_individual(past_data, 'ball1', top_ball1)
    ball6_candidates = predict_ball_individual(past_data, 'ball6', top_ball6)

    # 조합 생성 (ball1 != ball6)
    pairs = []
    for ball1, score1 in ball1_candidates:
        for ball6, score6 in ball6_candidates:
            if ball1 != ball6:  # 같은 번호 제외
                combined_score = score1 + score6
                pairs.append((ball1, ball6, combined_score))

    # 점수 높은 순 정렬
    pairs.sort(key=lambda x: -x[2])
    return pairs


def get_valid_ball_pairs() -> List[Tuple[int, int]]:
    """유효한 (ball1, ball6) 쌍 목록 생성 (45*44 = 1980개)"""
    pairs = []
    for ball1 in range(1, 46):
        for ball6 in range(1, 46):
            if ball1 != ball6:
                pairs.append((ball1, ball6))
    return pairs


def predict_ball1ball6(
    target_round: int,
    all_data: List[Dict],
    top_k: int = 400
) -> List[Tuple[int, int, float]]:
    """
    (ball1, ball6) 쌍 예측

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        top_k: 상위 K개 반환

    Returns:
        [(ball1, ball6, score), ...] 점수 높은 순
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    recent_10 = past_data[-10:]

    # ball1, ball6 빈도 계산
    freq_ball1 = Counter(d['ball1'] for d in recent_10)
    freq_ball6 = Counter(d['ball6'] for d in recent_10)

    # 모든 유효 쌍에 대해 점수 계산
    valid_pairs = get_valid_ball_pairs()
    scored_pairs = []

    for ball1, ball6 in valid_pairs:
        freq_score = freq_ball1.get(ball1, 0) + freq_ball6.get(ball6, 0)
        scored_pairs.append((ball1, ball6, freq_score))

    # 점수 높은 순 정렬
    scored_pairs.sort(key=lambda x: -x[2])

    return scored_pairs[:top_k]


if __name__ == '__main__':
    # 테스트
    data = load_winning_data()
    target = 1200

    print(f"=== {target}회차 (ball1, ball6) 예측 ===\n")

    pairs = predict_ball1ball6_v2(target, data, top_ball1=15, top_ball6=15)
    print("Top-20:")
    for i, (b1, b6, score) in enumerate(pairs[:20], 1):
        print(f"  {i:2d}. (ball1={b1:2d}, ball6={b6:2d}) score={score:.1f}")

    # 실제 당첨번호 확인
    actual = next((d for d in data if d['round'] == target), None)
    if actual:
        print(f"\n실제: ball1={actual['ball1']}, ball6={actual['ball6']}")
