"""
3_predict/common.py - 공통 유틸리티

- 데이터 로딩
- 하이브리드 클러스터링 (구간 + 빈도 가중치)
- 이상치 점수 계산
- 다양성 필터링 (3자리 이상 중복 제거)
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
from dataclasses import dataclass

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
DESCRIBE_PATH = BASE_DIR / "2_describe" / "describe.csv"

# 소수 집합
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# 5개 클러스터 정의 (하이브리드)
BASE_CLUSTERS = {
    'A': list(range(1, 10)),    # 1-9
    'B': list(range(10, 19)),   # 10-18
    'C': list(range(19, 28)),   # 19-27
    'D': list(range(28, 37)),   # 28-36
    'E': list(range(37, 46)),   # 37-45
}

# 클러스터 분배 패턴 (합계 = 6)
DISTRIBUTION_PATTERNS = [
    (1, 1, 1, 2, 1),  # 균등 + D 강조
    (1, 2, 1, 1, 1),  # 균등 + B 강조
    (1, 1, 2, 1, 1),  # 균등 + C 강조
    (0, 1, 2, 2, 1),  # A 생략
    (1, 1, 1, 1, 2),  # 균등 + E 강조
    (2, 1, 1, 1, 1),  # 균등 + A 강조
    (0, 2, 1, 2, 1),  # A 생략, B+D 강조
    (1, 1, 2, 2, 0),  # E 생략
    (1, 2, 2, 1, 0),  # E 생략, B+C 강조
    (0, 1, 1, 2, 2),  # A 생략, D+E 강조
]


@dataclass
class Combination:
    """6개 번호 조합"""
    numbers: Tuple[int, ...]  # (ord1, ord2, ord3, ord4, ord5, ord6)
    score: float = 0.0        # 이상치 점수 (낮을수록 좋음)
    cluster_pattern: str = "" # 클러스터 분포 패턴

    def __hash__(self):
        return hash(self.numbers)

    def __eq__(self, other):
        return self.numbers == other.numbers


def load_winning_data() -> List[Dict]:
    """당첨번호 데이터 로드 (o1~o45 포함)"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {
                'round': int(row['round']),
                'ord1': int(row['ord1']),
                'ord2': int(row['ord2']),
                'ord3': int(row['ord3']),
                'ord4': int(row['ord4']),
                'ord5': int(row['ord5']),
                'ord6': int(row['ord6']),
                'ord_bonus': int(row['ord_bonus']),
            }
            # ball1~ball6 추가
            for i in range(1, 7):
                data[f'ball{i}'] = int(row[f'ball{i}'])
            # o1~o45 추가 (Ball→Ord 변환용)
            for i in range(1, 46):
                data[f'o{i}'] = int(row[f'o{i}'])
            results.append(data)
    return sorted(results, key=lambda x: x['round'])


def load_describe_data() -> List[Dict]:
    """describe.csv 로드 (115개 컬럼)"""
    results = []
    with open(DESCRIBE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 숫자 컬럼 변환
            data = {}
            for k, v in row.items():
                if v == '':
                    data[k] = None
                elif v in ('True', 'False'):
                    data[k] = v == 'True'
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


def get_cluster(num: int) -> str:
    """번호가 속한 클러스터 반환"""
    if 1 <= num <= 9:
        return 'A'
    elif 10 <= num <= 18:
        return 'B'
    elif 19 <= num <= 27:
        return 'C'
    elif 28 <= num <= 36:
        return 'D'
    else:  # 37-45
        return 'E'


def get_cluster_weights(recent_data: List[Dict]) -> Dict[str, float]:
    """최근 10회차 기준 클러스터별 가중치 계산"""
    freq = Counter()
    for row in recent_data[-10:]:
        for i in range(1, 7):
            freq[row[f'ord{i}']] += 1

    weights = {}
    for cluster, nums in BASE_CLUSTERS.items():
        cluster_freq = sum(freq.get(n, 0) for n in nums)
        # 정규화: 9개 번호 × 10회차 × 6/45 = 12가 기대값
        weights[cluster] = cluster_freq / 12.0

    return weights


def get_cluster_pattern(numbers: Tuple[int, ...]) -> str:
    """조합의 클러스터 분포 패턴 반환"""
    clusters = [get_cluster(n) for n in numbers]
    pattern = Counter(clusters)
    return ''.join(f"{pattern.get(c, 0)}" for c in 'ABCDE')


def calculate_outlier_score(numbers: Tuple[int, ...], prev_numbers: Tuple[int, ...] = None) -> float:
    """
    이상치 점수 계산 (낮을수록 좋음)

    각 항목별 가중치:
    - 연속수 0~1개: 정상 (91.0%)
    - 소수 1~3개: 정상 (83.6%)
    - 홀수 2~4개: 정상 (84.2%)
    - 이월수 0~1개: 정상 (80.5%)
    - AC값 7~10: 정상 (83.1%)
    - 합계 100~159: 정상 (69.9%)
    - 동끝수 0~1쌍: 정상 (79.2%)
    """
    score = 0.0
    nums = list(numbers)

    # 1. 연속수 검사
    consecutive_count = 0
    for i in range(len(nums) - 1):
        if nums[i + 1] - nums[i] == 1:
            consecutive_count += 1
    if consecutive_count > 1:
        score += 1.0

    # 2. 소수 검사
    prime_count = sum(1 for n in nums if n in PRIMES)
    if prime_count < 1 or prime_count > 3:
        score += 0.9

    # 3. 홀짝 검사
    odd_count = sum(1 for n in nums if n % 2 == 1)
    if odd_count < 2 or odd_count > 4:
        score += 0.9

    # 4. 이월수 검사 (이전 회차 정보 필요)
    if prev_numbers:
        carryover = len(set(nums) & set(prev_numbers))
        if carryover > 1:
            score += 0.8

    # 5. AC값 검사
    diffs = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            diffs.add(abs(nums[i] - nums[j]))
    ac_value = len(diffs) - 5
    if ac_value < 7 or ac_value > 10:
        score += 0.8

    # 6. 합계 검사
    total = sum(nums)
    if total < 100 or total > 159:
        score += 0.7

    # 7. 동일 끝수 검사
    last_digits = [n % 10 for n in nums]
    digit_counts = Counter(last_digits)
    same_digit_pairs = sum(1 for c in digit_counts.values() if c >= 2)
    if same_digit_pairs > 1:
        score += 0.8

    return score


def is_diverse(new_combo: Tuple[int, ...], selected: List[Tuple[int, ...]]) -> bool:
    """새 조합이 기존 조합들과 3자리 이상 중복되는지 확인"""
    new_set = set(new_combo)
    for existing in selected:
        overlap = len(new_set & set(existing))
        if overlap >= 3:
            return False
    return True


def select_diverse_combinations(
    all_combos: List[Combination],
    max_count: int = 100
) -> List[Combination]:
    """
    다양성 필터링으로 조합 선택
    - 이상치 점수 낮은 순 정렬
    - 3자리 이상 중복 제거
    """
    # 점수순 정렬
    sorted_combos = sorted(all_combos, key=lambda x: x.score)

    selected = []
    selected_numbers = []

    for combo in sorted_combos:
        if is_diverse(combo.numbers, selected_numbers):
            selected.append(combo)
            selected_numbers.append(combo.numbers)
            if len(selected) >= max_count:
                break

    return selected


def get_hot_numbers(recent_data: List[Dict], top_n: int = 15) -> List[int]:
    """최근 10회차에서 가장 많이 나온 번호"""
    freq = Counter()
    for row in recent_data[-10:]:
        for i in range(1, 7):
            freq[row[f'ord{i}']] += 1
    return [n for n, _ in freq.most_common(top_n)]


def get_cold_numbers(recent_data: List[Dict], top_n: int = 10) -> List[int]:
    """최근 10회차에서 가장 적게 나온 번호"""
    freq = Counter()
    for row in recent_data[-10:]:
        for i in range(1, 7):
            freq[row[f'ord{i}']] += 1

    # 빈도 0인 번호도 포함
    for n in range(1, 46):
        if n not in freq:
            freq[n] = 0

    return [n for n, _ in freq.most_common()[-top_n:]]


def validate_combination(numbers: Tuple[int, ...]) -> bool:
    """조합 유효성 검사"""
    if len(numbers) != 6:
        return False
    if len(set(numbers)) != 6:
        return False
    if not all(1 <= n <= 45 for n in numbers):
        return False
    # 정렬 확인
    return list(numbers) == sorted(numbers)


def format_combination(numbers: Tuple[int, ...]) -> str:
    """조합을 문자열로 포맷"""
    return f"({', '.join(map(str, numbers))})"


def compare_with_actual(
    predicted: List[Combination],
    actual: Tuple[int, ...]
) -> Dict:
    """예측 조합들과 실제 당첨번호 비교"""
    actual_set = set(actual)

    best_match = 0
    best_combo = None
    matches_by_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for combo in predicted:
        match_count = len(set(combo.numbers) & actual_set)
        matches_by_count[match_count] += 1

        if match_count > best_match:
            best_match = match_count
            best_combo = combo

    return {
        'actual': actual,
        'best_match_count': best_match,
        'best_combo': best_combo,
        'distribution': matches_by_count,
        'is_perfect': best_match == 6
    }
