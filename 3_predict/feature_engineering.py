"""
3_predict/feature_engineering.py - 피처 엔지니어링

describe.csv의 115개 컬럼을 활용한 ML 피처 생성
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

BASE_DIR = Path(__file__).parent.parent
DESCRIBE_PATH = BASE_DIR / "2_describe" / "describe.csv"
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"

# 소수 집합
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_describe_data() -> List[Dict]:
    """describe.csv 로드"""
    results = []
    with open(DESCRIBE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {}
            for k, v in row.items():
                if v == '':
                    data[k] = None
                elif v in ('True', 'False'):
                    data[k] = 1 if v == 'True' else 0
                else:
                    try:
                        if '.' in v:
                            data[k] = float(v)
                        else:
                            data[k] = int(v)
                    except ValueError:
                        data[k] = v
            results.append(data)
    return results


def build_ord1ord6_features(
    target_round: int,
    all_data: List[Dict],
    ord1: int,
    ord6: int
) -> Dict:
    """
    (ord1, ord6) 쌍에 대한 피처 생성

    피처 목록:
    - span, ord1, ord6 기본값
    - 최근 10회차 빈도 통계
    - 구간 분포
    - 이전 회차 패턴
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data
    recent_20 = past_data[-20:] if len(past_data) >= 20 else past_data

    span = ord6 - ord1

    # ord1, ord6 빈도
    freq_ord1 = Counter(d['ord1'] for d in recent_10)
    freq_ord6 = Counter(d['ord6'] for d in recent_10)
    freq_ord1_20 = Counter(d['ord1'] for d in recent_20)
    freq_ord6_20 = Counter(d['ord6'] for d in recent_20)

    # span 빈도
    span_freq = Counter(d['ord6'] - d['ord1'] for d in recent_10)

    # 구간 정보
    def get_range(n):
        if n <= 9: return 0
        elif n <= 18: return 1
        elif n <= 27: return 2
        elif n <= 36: return 3
        else: return 4

    # 이전 회차 정보
    prev = past_data[-1] if past_data else None

    features = {
        # 기본 정보
        'ord1': ord1,
        'ord6': ord6,
        'span': span,

        # 빈도 정보
        'ord1_freq_10': freq_ord1.get(ord1, 0),
        'ord6_freq_10': freq_ord6.get(ord6, 0),
        'ord1_freq_20': freq_ord1_20.get(ord1, 0),
        'ord6_freq_20': freq_ord6_20.get(ord6, 0),
        'span_freq_10': span_freq.get(span, 0),

        # 구간 정보
        'ord1_range': get_range(ord1),
        'ord6_range': get_range(ord6),

        # 소수/홀짝
        'ord1_is_prime': 1 if ord1 in PRIMES else 0,
        'ord6_is_prime': 1 if ord6 in PRIMES else 0,
        'ord1_is_odd': ord1 % 2,
        'ord6_is_odd': ord6 % 2,

        # 끝자리
        'ord1_last_digit': ord1 % 10,
        'ord6_last_digit': ord6 % 10,

        # 이전 회차 대비
        'ord1_diff_prev': ord1 - prev['ord1'] if prev else 0,
        'ord6_diff_prev': ord6 - prev['ord6'] if prev else 0,
        'ord1_in_prev': 1 if prev and ord1 in [prev[f'ord{i}'] for i in range(1, 7)] else 0,
        'ord6_in_prev': 1 if prev and ord6 in [prev[f'ord{i}'] for i in range(1, 7)] else 0,

        # span 통계
        'span_normalized': span / 44,  # 최대 span = 44
        'ideal_gap': span / 5,
    }

    return features


def build_ordN_features(
    target_round: int,
    all_data: List[Dict],
    ord1: int,
    ord6: int,
    position: int,  # 2, 3, 4, 5
    known_ords: Dict[int, int] = None  # 이미 결정된 ord 값들
) -> Dict:
    """
    ord2~ord5에 대한 피처 생성

    position: 예측할 위치 (2, 3, 4, 5)
    known_ords: {2: 값, 3: 값, ...} 이미 결정된 값
    """
    past_data = [d for d in all_data if d['round'] < target_round]
    recent_10 = past_data[-10:] if len(past_data) >= 10 else past_data

    span = ord6 - ord1
    known_ords = known_ords or {}

    # 해당 위치의 빈도
    freq = Counter(d[f'ord{position}'] for d in recent_10)

    # 위치별 기본 비율 (과거 데이터 기반)
    ratios = []
    for d in past_data[-50:]:
        d_span = d['ord6'] - d['ord1']
        if d_span > 0:
            ratio = (d[f'ord{position}'] - d['ord1']) / d_span
            ratios.append(ratio)
    avg_ratio = np.mean(ratios) if ratios else 0.5

    # 구간 선호도 계산
    range_counts = Counter()
    for d in past_data:
        val = d[f'ord{position}']
        if val <= 9: range_counts[0] += 1
        elif val <= 18: range_counts[1] += 1
        elif val <= 27: range_counts[2] += 1
        elif val <= 36: range_counts[3] += 1
        else: range_counts[4] += 1

    total = sum(range_counts.values()) or 1
    range_probs = [range_counts[i] / total for i in range(5)]

    prev = past_data[-1] if past_data else None

    features = {
        'position': position,
        'ord1': ord1,
        'ord6': ord6,
        'span': span,

        # 위치별 통계
        'avg_ratio': avg_ratio,
        'expected_value': ord1 + span * avg_ratio,

        # 범위 정보
        'range_prob_A': range_probs[0],
        'range_prob_B': range_probs[1],
        'range_prob_C': range_probs[2],
        'range_prob_D': range_probs[3],
        'range_prob_E': range_probs[4],

        # 이전 회차 정보
        'prev_value': prev[f'ord{position}'] if prev else 0,
    }

    # 알려진 ord 값 추가
    for pos, val in known_ords.items():
        features[f'known_ord{pos}'] = val

    return features


def build_training_data_ord1ord6(all_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    (ord1, ord6) 예측을 위한 학습 데이터 생성

    Returns:
        X: 피처 배열
        y: 정답 (ord1 * 100 + ord6) 인코딩
    """
    X_list = []
    y_list = []

    for i in range(10, len(all_data)):
        row = all_data[i]
        target_round = row['round']
        actual_ord1 = row['ord1']
        actual_ord6 = row['ord6']

        # 정답 쌍의 피처 생성
        features = build_ord1ord6_features(target_round, all_data, actual_ord1, actual_ord6)

        feature_values = [
            features['span'],
            features['ord1_freq_10'],
            features['ord6_freq_10'],
            features['ord1_freq_20'],
            features['ord6_freq_20'],
            features['span_freq_10'],
            features['ord1_range'],
            features['ord6_range'],
            features['ord1_is_prime'],
            features['ord6_is_prime'],
            features['ord1_is_odd'],
            features['ord6_is_odd'],
            features['ord1_last_digit'],
            features['ord6_last_digit'],
            features['ord1_diff_prev'],
            features['ord6_diff_prev'],
            features['span_normalized'],
        ]

        X_list.append(feature_values)
        y_list.append(1)  # 정답

        # Negative samples (랜덤 오답 쌍 5개)
        import random
        for _ in range(5):
            rand_ord1 = random.randint(1, 40)
            rand_ord6 = random.randint(rand_ord1 + 5, 45)
            if (rand_ord1, rand_ord6) != (actual_ord1, actual_ord6):
                neg_features = build_ord1ord6_features(target_round, all_data, rand_ord1, rand_ord6)
                neg_values = [
                    neg_features['span'],
                    neg_features['ord1_freq_10'],
                    neg_features['ord6_freq_10'],
                    neg_features['ord1_freq_20'],
                    neg_features['ord6_freq_20'],
                    neg_features['span_freq_10'],
                    neg_features['ord1_range'],
                    neg_features['ord6_range'],
                    neg_features['ord1_is_prime'],
                    neg_features['ord6_is_prime'],
                    neg_features['ord1_is_odd'],
                    neg_features['ord6_is_odd'],
                    neg_features['ord1_last_digit'],
                    neg_features['ord6_last_digit'],
                    neg_features['ord1_diff_prev'],
                    neg_features['ord6_diff_prev'],
                    neg_features['span_normalized'],
                ]
                X_list.append(neg_values)
                y_list.append(0)  # 오답

    return np.array(X_list), np.array(y_list)


def build_training_data_ordN(all_data: List[Dict], position: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    ord2~ord5 예측을 위한 학습 데이터 생성

    position: 2, 3, 4, 5
    Returns:
        X: 피처 배열
        y: 실제 ord 값
    """
    X_list = []
    y_list = []

    for i in range(10, len(all_data)):
        row = all_data[i]
        target_round = row['round']

        ord1 = row['ord1']
        ord6 = row['ord6']
        actual_value = row[f'ord{position}']

        # known_ords 구성 (position 이전의 값들)
        known_ords = {}
        if position > 2:
            known_ords[2] = row['ord2']
        if position > 3:
            known_ords[3] = row['ord3']
        if position > 4:
            known_ords[4] = row['ord4']

        features = build_ordN_features(target_round, all_data, ord1, ord6, position, known_ords)

        feature_values = [
            features['span'],
            features['avg_ratio'],
            features['expected_value'],
            features['range_prob_A'],
            features['range_prob_B'],
            features['range_prob_C'],
            features['range_prob_D'],
            features['range_prob_E'],
            features['prev_value'],
            ord1,
            ord6,
        ]

        # known_ords 추가
        for pos in [2, 3, 4]:
            if pos < position:
                feature_values.append(known_ords.get(pos, 0))
            else:
                feature_values.append(0)

        X_list.append(feature_values)
        y_list.append(actual_value)

    return np.array(X_list), np.array(y_list)


if __name__ == '__main__':
    from common import load_winning_data

    data = load_winning_data()
    print(f"총 {len(data)}개 회차 로드")

    # ord1ord6 학습 데이터
    X, y = build_training_data_ord1ord6(data)
    print(f"\nord1ord6 학습 데이터: X={X.shape}, y={y.shape}")
    print(f"  정답 비율: {y.mean():.2%}")

    # ord4 학습 데이터
    X4, y4 = build_training_data_ordN(data, 4)
    print(f"\nord4 학습 데이터: X={X4.shape}, y={y4.shape}")
    print(f"  ord4 범위: {y4.min()} ~ {y4.max()}")
