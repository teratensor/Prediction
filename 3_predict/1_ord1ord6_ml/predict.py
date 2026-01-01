"""
1_ord1ord6_ml/predict.py - (ord1, ord6) 쌍 예측

ML 앙상블 방식:
- XGBoost (70%) + 빈도 기반 (30%)
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pickle

# 상위 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_winning_data, get_cluster_weights, BASE_CLUSTERS
from feature_engineering import build_ord1ord6_features

MODEL_PATH = Path(__file__).parent / "model.pkl"

def _load_xgb_model():
    """XGBoost 모델 로드"""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

_XGB_MODEL = None

def get_xgb_model():
    """싱글톤 XGBoost 모델 반환"""
    global _XGB_MODEL
    if _XGB_MODEL is None:
        _XGB_MODEL = _load_xgb_model()
    return _XGB_MODEL


def get_valid_pairs() -> List[Tuple[int, int]]:
    """유효한 (ord1, ord6) 쌍 목록 생성 (820개)"""
    pairs = []
    for ord1 in range(1, 41):  # ord1: 1~40
        for ord6 in range(ord1 + 5, 46):  # ord6: ord1+5 ~ 45 (최소 5칸 차이)
            pairs.append((ord1, ord6))
    return pairs


def predict_ord1_individual(
    past_data: List[Dict],
    top_k: int = 20
) -> List[Tuple[int, float]]:
    """
    ord1 개별 예측 (범위 가중치 + 빈도 기반)

    ord1 분포 (378회차 분석):
    - 1-5: 38.4%
    - 6-10: 36.5%
    - 11-15: 18.5%
    - 16-20: 5.6%
    - 21+: 1.1%

    Returns:
        [(ord1, score), ...] 점수 높은 순
    """
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    recent_20 = past_data[-20:] if len(past_data) >= 20 else past_data

    freq_10 = Counter(d['ord1'] for d in recent_10)
    freq_20 = Counter(d['ord1'] for d in recent_20)

    candidates = []
    for ord1 in range(1, 41):
        # 빈도 점수
        freq_score = freq_10.get(ord1, 0) * 2 + freq_20.get(ord1, 0)

        # 범위 가중치 (실제 분포 반영)
        if 1 <= ord1 <= 5:
            range_weight = 3.0  # 38.4%
        elif 6 <= ord1 <= 10:
            range_weight = 2.8  # 36.5%
        elif 11 <= ord1 <= 15:
            range_weight = 1.5  # 18.5%
        elif 16 <= ord1 <= 20:
            range_weight = 0.5  # 5.6%
        else:
            range_weight = 0.1  # 1.1%

        total_score = freq_score + range_weight
        candidates.append((ord1, total_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def predict_ord6_individual(
    past_data: List[Dict],
    top_k: int = 20
) -> List[Tuple[int, float]]:
    """
    ord6 개별 예측 (범위 가중치 + 빈도 기반)

    ord6 분포 (378회차 분석):
    - 43-45: 31.4%
    - 40-42: 22.4%
    - 35-39: 27.4%
    - 30-34: 13.8%
    - 25-29: 4.2%
    - <25: 0.8%

    Returns:
        [(ord6, score), ...] 점수 높은 순
    """
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    recent_20 = past_data[-20:] if len(past_data) >= 20 else past_data

    freq_10 = Counter(d['ord6'] for d in recent_10)
    freq_20 = Counter(d['ord6'] for d in recent_20)

    candidates = []
    for ord6 in range(6, 46):
        # 빈도 점수
        freq_score = freq_10.get(ord6, 0) * 2 + freq_20.get(ord6, 0)

        # 범위 가중치 (실제 분포 반영)
        if 43 <= ord6 <= 45:
            range_weight = 3.0  # 31.4%
        elif 40 <= ord6 <= 42:
            range_weight = 2.2  # 22.4%
        elif 35 <= ord6 <= 39:
            range_weight = 2.5  # 27.4%
        elif 30 <= ord6 <= 34:
            range_weight = 1.2  # 13.8%
        elif 25 <= ord6 <= 29:
            range_weight = 0.4  # 4.2%
        else:
            range_weight = 0.1  # 0.8%

        total_score = freq_score + range_weight
        candidates.append((ord6, total_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates[:top_k]


def predict_ord1ord6_v2(
    target_round: int,
    all_data: List[Dict],
    top_ord1: int = 20,
    top_ord6: int = 20
) -> List[Tuple[int, int, float]]:
    """
    개선된 (ord1, ord6) 쌍 예측 - 개별 예측 후 조합

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        top_ord1: ord1 상위 K개
        top_ord6: ord6 상위 K개

    Returns:
        [(ord1, ord6, score), ...] 최대 top_ord1 × top_ord6 개
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    # 개별 예측
    ord1_candidates = predict_ord1_individual(past_data, top_ord1)
    ord6_candidates = predict_ord6_individual(past_data, top_ord6)

    # 조합 생성 (유효한 쌍만)
    pairs = []
    for ord1, score1 in ord1_candidates:
        for ord6, score6 in ord6_candidates:
            if ord6 >= ord1 + 5:  # 최소 5칸 차이
                combined_score = score1 + score6
                pairs.append((ord1, ord6, combined_score))

    # 점수 높은 순 정렬
    pairs.sort(key=lambda x: -x[2])
    return pairs


def calculate_pair_score_ml(
    ord1: int,
    ord6: int,
    target_round: int,
    all_data: List[Dict],
    freq_ord1: Counter,
    freq_ord6: Counter,
    span_freq: Counter = None
) -> float:
    """
    ML 앙상블 기반 (ord1, ord6) 쌍 점수 계산

    점수 = 빈도점수(60%) + span점수(20%) + 다양성(20%)
    """
    # 빈도 점수 (가장 중요)
    freq_score = (freq_ord1.get(ord1, 0) + freq_ord6.get(ord6, 0)) / 20.0

    # span 점수
    span = ord6 - ord1
    if span_freq:
        span_score = span_freq.get(span, 0) / 10.0
    else:
        span_score = 0.0

    # 다양성 점수: ord1/ord6이 다양한 범위에 분포하도록
    # 모든 범위가 골고루 선택되도록 가중치 부여
    range_diversity = 0.0
    if 1 <= ord1 <= 10:
        range_diversity += 0.15  # 낮은 ord1
    elif 11 <= ord1 <= 20:
        range_diversity += 0.25  # 중간 ord1 (더 흔함)
    elif 21 <= ord1 <= 30:
        range_diversity += 0.20  # 높은 ord1

    if 35 <= ord6 <= 40:
        range_diversity += 0.15  # 중간 ord6
    elif 41 <= ord6 <= 45:
        range_diversity += 0.25  # 높은 ord6 (더 흔함)

    # 앙상블 점수
    total_score = freq_score * 0.6 + span_score * 0.2 + range_diversity * 0.2

    return total_score


def calculate_pair_score(
    ord1: int,
    ord6: int,
    freq_ord1: Counter,
    freq_ord6: Counter,
    span_freq: Counter,
    cluster_weights: Dict[str, float]
) -> float:
    """
    (ord1, ord6) 쌍의 점수 계산 (레거시, 휴리스틱만)

    점수 = 빈도점수(70%) + span점수(20%) + 클러스터가중치(10%)
    """
    # 1. 빈도 점수 (ord1, ord6 각각의 출현 빈도)
    freq_score = (freq_ord1.get(ord1, 0) + freq_ord6.get(ord6, 0)) / 20.0

    # 2. span 점수
    span = ord6 - ord1
    span_score = span_freq.get(span, 0) / 10.0

    # 3. 클러스터 가중치
    cluster_ord1 = 'A' if ord1 <= 9 else ('B' if ord1 <= 18 else ('C' if ord1 <= 27 else ('D' if ord1 <= 36 else 'E')))
    cluster_ord6 = 'A' if ord6 <= 9 else ('B' if ord6 <= 18 else ('C' if ord6 <= 27 else ('D' if ord6 <= 36 else 'E')))
    cluster_score = (cluster_weights.get(cluster_ord1, 1.0) + cluster_weights.get(cluster_ord6, 1.0)) / 2.0

    # 종합 점수
    total_score = freq_score * 0.7 + span_score * 0.2 + cluster_score * 0.1

    return total_score


def predict_ord1ord6(
    target_round: int,
    all_data: List[Dict],
    top_k: int = 100,
    use_ml: bool = True
) -> List[Tuple[int, int, float]]:
    """
    (ord1, ord6) 쌍 예측 (ML 앙상블 기반)

    Args:
        target_round: 예측할 회차
        all_data: 전체 데이터
        top_k: 상위 K개 반환
        use_ml: ML 모델 사용 여부

    Returns:
        [(ord1, ord6, score), ...] 점수 높은 순
    """
    # 과거 데이터만 사용
    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    recent_10 = past_data[-10:]

    # ord1, ord6 빈도 계산
    freq_ord1 = Counter(d['ord1'] for d in recent_10)
    freq_ord6 = Counter(d['ord6'] for d in recent_10)

    # 모든 유효 쌍에 대해 점수 계산
    valid_pairs = get_valid_pairs()
    scored_pairs = []

    span_freq = Counter(d['ord6'] - d['ord1'] for d in recent_10)

    if use_ml:
        # 빈도 + 다양성 기반 방식
        for ord1, ord6 in valid_pairs:
            score = calculate_pair_score_ml(
                ord1, ord6, target_round, all_data, freq_ord1, freq_ord6, span_freq
            )
            scored_pairs.append((ord1, ord6, score))
    else:
        # 레거시 휴리스틱 방식
        span_freq = Counter(d['ord6'] - d['ord1'] for d in recent_10)
        cluster_weights = get_cluster_weights(past_data)
        for ord1, ord6 in valid_pairs:
            score = calculate_pair_score(
                ord1, ord6, freq_ord1, freq_ord6, span_freq, cluster_weights
            )
            scored_pairs.append((ord1, ord6, score))

    # 점수 높은 순 정렬
    scored_pairs.sort(key=lambda x: -x[2])

    return scored_pairs[:top_k]


def get_frequency_filtered_pairs(
    target_round: int,
    all_data: List[Dict],
    min_freq: int = 1,
    top_k: int = 200
) -> List[Tuple[int, int, float]]:
    """
    다양성 기반 쌍 반환

    전략: 모든 유효한 쌍 (820개) 중에서
    - 빈도 기반 상위 30%
    - (ord1 범위, ord6 범위) 그리드에서 균등 선택 40%
    - 랜덤 선택 30%
    """
    import random
    random.seed(target_round)  # 재현 가능한 랜덤

    past_data = [d for d in all_data if d['round'] < target_round]
    if len(past_data) < 10:
        return []

    recent_10 = past_data[-10:]
    recent_20 = past_data[-20:]

    # 빈도 계산 (10회, 20회)
    freq_ord1_10 = Counter(d['ord1'] for d in recent_10)
    freq_ord6_10 = Counter(d['ord6'] for d in recent_10)
    freq_ord1_20 = Counter(d['ord1'] for d in recent_20)
    freq_ord6_20 = Counter(d['ord6'] for d in recent_20)

    # 모든 유효 쌍에 대해 점수 계산
    all_pairs = []
    for ord1 in range(1, 41):
        for ord6 in range(ord1 + 5, 46):
            # 빈도 점수 (10회 + 20회)
            freq_score = (
                freq_ord1_10.get(ord1, 0) * 2 +
                freq_ord6_10.get(ord6, 0) * 2 +
                freq_ord1_20.get(ord1, 0) +
                freq_ord6_20.get(ord6, 0)
            ) / 30.0

            all_pairs.append((ord1, ord6, freq_score))

    result = []
    existing = set()

    # 1. 빈도 기반 상위 30%
    freq_count = int(top_k * 0.3)
    sorted_by_freq = sorted(all_pairs, key=lambda x: -x[2])
    for p in sorted_by_freq[:freq_count]:
        if (p[0], p[1]) not in existing:
            result.append(p)
            existing.add((p[0], p[1]))

    # 2. 그리드 균등 선택 40%
    grid_count = int(top_k * 0.4)
    ord1_ranges = [(1, 10), (11, 20), (21, 30), (31, 40)]
    ord6_ranges = [(20, 32), (33, 38), (39, 42), (43, 45)]
    per_cell = max(2, grid_count // 16)

    for r1_min, r1_max in ord1_ranges:
        for r6_min, r6_max in ord6_ranges:
            cell_pairs = [
                p for p in all_pairs
                if r1_min <= p[0] <= r1_max and r6_min <= p[1] <= r6_max
                and (p[0], p[1]) not in existing
            ]
            # 빈도 높은 순으로 선택
            cell_pairs.sort(key=lambda x: -x[2])
            for p in cell_pairs[:per_cell]:
                result.append(p)
                existing.add((p[0], p[1]))

    # 3. 랜덤 선택 30%
    random_count = top_k - len(result)
    remaining = [p for p in all_pairs if (p[0], p[1]) not in existing]
    random.shuffle(remaining)
    for p in remaining[:random_count]:
        result.append(p)
        existing.add((p[0], p[1]))

    return result[:top_k]


if __name__ == '__main__':
    # 테스트
    data = load_winning_data()
    target = 1200

    print(f"=== {target}회차 (ord1, ord6) 예측 ===\n")

    # 방법 1: 전체 점수 기반
    pairs = predict_ord1ord6(target, data, top_k=20)
    print("Top-20 (점수 기반):")
    for i, (o1, o6, score) in enumerate(pairs[:10], 1):
        print(f"  {i:2d}. ({o1:2d}, {o6:2d}) span={o6-o1:2d} score={score:.3f}")

    # 방법 2: 빈도 필터링
    print("\nTop-20 (빈도 필터링, min_freq=2):")
    freq_pairs = get_frequency_filtered_pairs(target, data, min_freq=2, top_k=20)
    for i, (o1, o6, score) in enumerate(freq_pairs[:10], 1):
        print(f"  {i:2d}. ({o1:2d}, {o6:2d}) span={o6-o1:2d} score={score:.1f}")

    # 실제 당첨번호 확인
    actual = next((d for d in data if d['round'] == target), None)
    if actual:
        print(f"\n실제: ord1={actual['ord1']}, ord6={actual['ord6']}")
        actual_pair = (actual['ord1'], actual['ord6'])

        # 순위 확인
        for i, (o1, o6, _) in enumerate(pairs):
            if (o1, o6) == actual_pair:
                print(f"점수 기반 순위: {i+1}위")
                break

        for i, (o1, o6, _) in enumerate(freq_pairs):
            if (o1, o6) == actual_pair:
                print(f"빈도 필터링 순위: {i+1}위")
                break
