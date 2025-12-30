"""
8개 인사이트 기반 ord2 예측 피처 추출

각 인사이트에서 ord2 예측에 필요한 피처를 추출
"""

import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH.parent / "1_data" / "winning_numbers.csv"

# 소수 목록
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# 범위 정의
RANGES = [(1, 9), (10, 19), (20, 29), (30, 39), (40, 45)]


def load_winning_numbers() -> List[Dict]:
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'ord1': int(row['ord1']),
                'ord2': int(row['ord2']),
                'ord3': int(row['ord3']),
                'ord4': int(row['ord4']),
                'ord5': int(row['ord5']),
                'ord6': int(row['ord6']),
                'balls': sorted([int(row[f'ord{i}']) for i in range(1, 7)]),
            })
    return sorted(results, key=lambda x: x['round'])


def get_range_index(num: int) -> int:
    """번호가 속한 범위 인덱스 반환 (0-4)"""
    for i, (start, end) in enumerate(RANGES):
        if start <= num <= end:
            return i
    return 4


class FeatureExtractor:
    """8개 인사이트 기반 피처 추출기"""

    def __init__(self, train_data: List[Dict]):
        """
        Args:
            train_data: 학습용 당첨번호 데이터 (목표 회차 이전까지)
        """
        self.train_data = train_data
        self._compute_statistics()

    def _compute_statistics(self):
        """학습 데이터 기반 통계 계산"""
        # 1. 전체 번호 빈도
        self.ball_freq = Counter()
        for r in self.train_data:
            for b in r['balls']:
                self.ball_freq[b] += 1

        # 2. 포지션별 번호 빈도
        self.pos_freq = {i: Counter() for i in range(1, 7)}
        for r in self.train_data:
            for i in range(1, 7):
                self.pos_freq[i][r[f'ord{i}']] += 1

        # 3. ord2 범위별 빈도
        self.ord2_range_freq = Counter()
        for r in self.train_data:
            self.ord2_range_freq[get_range_index(r['ord2'])] += 1

        # 4. 연속수 통계
        self.consecutive_count = 0
        for r in self.train_data:
            if r['ord2'] == r['ord1'] + 1:
                self.consecutive_count += 1
        self.consecutive_prob = self.consecutive_count / len(self.train_data) if self.train_data else 0

        # 5. 합계 통계
        self.sum_stats = []
        for r in self.train_data:
            self.sum_stats.append(sum(r['balls']))
        self.avg_sum = sum(self.sum_stats) / len(self.sum_stats) if self.sum_stats else 139

        # 6. Top24/Mid14/Rest7 세그먼트 (최근 10회 기반)
        self._compute_segments()

        # 7. HOT/COLD 비트 (최근 10회 기반)
        self._compute_hot_cold()

    def _compute_segments(self):
        """Top24/Mid14/Rest7 세그먼트 계산 (최근 10회 기반)"""
        recent = self.train_data[-10:] if len(self.train_data) >= 10 else self.train_data
        freq = Counter()
        for r in recent:
            for b in r['balls']:
                freq[b] += 1

        sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))
        self.top24 = set(sorted_nums[:24])
        self.mid14 = set(sorted_nums[24:38])
        self.rest7 = set(sorted_nums[38:])

    def _compute_hot_cold(self):
        """HOT/COLD 비트 계산 (최근 10회 기반)"""
        recent = self.train_data[-10:] if len(self.train_data) >= 10 else self.train_data
        freq = Counter()
        for r in recent:
            for b in r['balls']:
                freq[b] += 1

        sorted_nums = sorted(range(1, 46), key=lambda x: -freq.get(x, 0))
        self.hot_bits = set(sorted_nums[:10])  # 상위 10개
        self.cold_bits = set(sorted_nums[-10:])  # 하위 10개

    def extract_features(self, ord1: int, ord6: int, candidate: int) -> Dict:
        """
        ord2 후보에 대한 피처 추출

        Args:
            ord1: 첫번째 번호
            ord6: 마지막 번호
            candidate: ord2 후보 번호

        Returns:
            피처 딕셔너리
        """
        features = {}

        # 기본 제약 조건 확인
        if candidate <= ord1 or candidate >= ord6:
            features['valid'] = 0
            return features

        features['valid'] = 1
        features['candidate'] = candidate

        # 1. 4_range: 범위 피처
        range_idx = get_range_index(candidate)
        features['range_idx'] = range_idx
        total = sum(self.ord2_range_freq.values())
        features['range_prob'] = self.ord2_range_freq.get(range_idx, 0) / total if total > 0 else 0

        # 2. 7_onehot: 포지션 빈도 피처
        total_pos2 = sum(self.pos_freq[2].values())
        features['pos2_freq'] = self.pos_freq[2].get(candidate, 0) / total_pos2 if total_pos2 > 0 else 0
        features['is_hot'] = 1 if candidate in self.hot_bits else 0
        features['is_cold'] = 1 if candidate in self.cold_bits else 0

        # 3. 3_sum: 합계 기여도
        # ord2가 전체 합계에 미치는 예상 영향
        remaining_positions = 4  # ord3, ord4, ord5 (ord1, ord6 제외)
        expected_remaining = self.avg_sum - ord1 - ord6 - candidate
        features['sum_contribution'] = candidate / self.avg_sum
        features['expected_remaining_avg'] = expected_remaining / remaining_positions if remaining_positions > 0 else 0

        # 4. 6_shortcode: 세그먼트 피처
        if candidate in self.top24:
            features['segment'] = 0  # Top24
            features['segment_name'] = 'top24'
        elif candidate in self.mid14:
            features['segment'] = 1  # Mid14
            features['segment_name'] = 'mid14'
        else:
            features['segment'] = 2  # Rest7
            features['segment_name'] = 'rest7'

        # 5. 5_prime: 소수 피처
        features['is_prime'] = 1 if candidate in PRIMES else 0

        # 6. 1_consecutive: 연속수 피처
        features['is_consecutive'] = 1 if candidate == ord1 + 1 else 0
        features['consecutive_prob'] = self.consecutive_prob

        # 7. 2_lastnum: 스팬 제약
        span = ord6 - ord1
        # ord2의 상대적 위치 (0~1 사이)
        features['relative_position'] = (candidate - ord1) / span if span > 0 else 0

        # 8. 8_ac: AC 기여도 (단순화)
        # AC는 전체 조합의 특성이므로 여기서는 간접적으로만 사용
        features['ac_potential'] = 1  # 추후 계산

        # 전체 빈도
        total_freq = sum(self.ball_freq.values())
        features['overall_freq'] = self.ball_freq.get(candidate, 0) / total_freq if total_freq > 0 else 0

        # 최근 출현 여부 (최근 3회)
        recent_3 = self.train_data[-3:] if len(self.train_data) >= 3 else self.train_data
        recent_balls = set()
        for r in recent_3:
            recent_balls.update(r['balls'])
        features['in_recent_3'] = 1 if candidate in recent_balls else 0

        return features

    def get_all_candidates(self, ord1: int, ord6: int) -> List[Dict]:
        """
        주어진 ord1, ord6에 대해 가능한 모든 ord2 후보와 피처 반환

        Args:
            ord1: 첫번째 번호
            ord6: 마지막 번호

        Returns:
            피처가 포함된 후보 리스트
        """
        candidates = []
        for num in range(ord1 + 1, ord6):
            features = self.extract_features(ord1, ord6, num)
            if features.get('valid', 0) == 1:
                candidates.append(features)
        return candidates


def create_training_dataset(data: List[Dict], min_train_size: int = 50) -> List[Tuple[Dict, int]]:
    """
    학습용 데이터셋 생성 (피처, 레이블)

    Rolling window 방식: 각 회차 예측 시 이전 회차들만 학습 데이터로 사용

    Args:
        data: 전체 당첨번호 데이터
        min_train_size: 최소 학습 데이터 크기

    Returns:
        (피처, 정답ord2) 튜플 리스트
    """
    dataset = []

    for i in range(min_train_size, len(data)):
        train_data = data[:i]
        target = data[i]

        extractor = FeatureExtractor(train_data)

        # 실제 ord1, ord6 사용
        ord1 = target['ord1']
        ord6 = target['ord6']
        actual_ord2 = target['ord2']

        # 모든 후보에 대한 피처 추출
        candidates = extractor.get_all_candidates(ord1, ord6)

        # 정답 후보 찾기
        for cand in candidates:
            is_correct = 1 if cand['candidate'] == actual_ord2 else 0
            dataset.append({
                'round': target['round'],
                'ord1': ord1,
                'ord6': ord6,
                'features': cand,
                'label': is_correct,
                'actual_ord2': actual_ord2
            })

    return dataset


if __name__ == "__main__":
    # 테스트
    data = load_winning_numbers()
    print(f"총 {len(data)}회차 데이터 로드")

    # 첫 100회차로 피처 추출 테스트
    train_data = data[:100]
    target = data[100]

    extractor = FeatureExtractor(train_data)

    print(f"\n목표 회차: {target['round']}")
    print(f"ord1={target['ord1']}, ord6={target['ord6']}, 실제 ord2={target['ord2']}")

    candidates = extractor.get_all_candidates(target['ord1'], target['ord6'])

    print(f"\n후보 수: {len(candidates)}")
    print("\n상위 5개 후보 (pos2_freq 기준):")

    sorted_candidates = sorted(candidates, key=lambda x: x['pos2_freq'], reverse=True)
    for i, cand in enumerate(sorted_candidates[:5]):
        print(f"  {i+1}. {cand['candidate']}: range={cand['range_idx']}, "
              f"pos2_freq={cand['pos2_freq']:.4f}, "
              f"is_hot={cand['is_hot']}, is_prime={cand['is_prime']}, "
              f"is_consecutive={cand['is_consecutive']}")

    # 실제 ord2 정보
    actual = next((c for c in candidates if c['candidate'] == target['ord2']), None)
    if actual:
        print(f"\n실제 ord2({target['ord2']}) 피처:")
        for k, v in actual.items():
            if k not in ['valid', 'segment_name']:
                print(f"  {k}: {v}")
