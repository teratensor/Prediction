"""
5_ball_predict/common.py - Ball 기반 예측 공통 유틸리티

Ball은 추첨 순서 (1~6번째로 나온 공)
Ord와 달리 정렬되지 않음
"""

import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
from dataclasses import dataclass

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"

# 소수 집합
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


@dataclass
class BallCombination:
    """6개 번호 조합 (ball 순서)"""
    numbers: Tuple[int, ...]  # (ball1, ball2, ball3, ball4, ball5, ball6)
    score: float = 0.0        # 이상치 점수 (낮을수록 좋음)

    def __hash__(self):
        return hash(self.numbers)

    def __eq__(self, other):
        return self.numbers == other.numbers

    def to_sorted(self) -> Tuple[int, ...]:
        """정렬된 번호 반환 (ord 형태)"""
        return tuple(sorted(self.numbers))


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


def ball_to_ord_position(ball_num: int, round_data: Dict) -> int:
    """
    Ball 번호를 해당 회차의 판매순위(Ord position)로 변환

    예: ball_num=3, round_data에서 o42=3이면 → 42 반환
    (3번 공의 판매순위가 42위)

    Args:
        ball_num: Ball 번호 (1~45)
        round_data: 해당 회차 데이터 (o1~o45 포함)

    Returns:
        판매순위 (1~45), 없으면 0
    """
    # o1~o45 중에서 값이 ball_num인 인덱스를 찾음
    for i in range(1, 46):
        if round_data.get(f'o{i}', 0) == ball_num:
            return i
    return 0


def convert_ball_combo_to_ord(
    ball_combo: Tuple[int, ...],
    round_data: Dict
) -> Tuple[int, ...]:
    """
    Ball 조합을 Ord 순서로 변환

    Args:
        ball_combo: Ball 조합 (ball1, ball2, ..., ball6)
        round_data: 해당 회차 데이터

    Returns:
        Ord 순서로 변환된 튜플
    """
    return tuple(ball_to_ord_position(b, round_data) for b in ball_combo)


def calculate_ball_outlier_score(numbers: Tuple[int, ...], prev_numbers: Tuple[int, ...] = None) -> float:
    """
    Ball 조합의 이상치 점수 계산 (낮을수록 좋음)
    정렬 후 평가
    """
    score = 0.0
    nums = sorted(list(numbers))

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

    # 4. 이월수 검사
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


def is_diverse_ball(new_combo: Tuple[int, ...], selected: List[Tuple[int, ...]], min_overlap: int = 4) -> bool:
    """새 조합이 기존 조합들과 min_overlap자리 이상 중복되는지 확인"""
    new_set = set(new_combo)
    for existing in selected:
        overlap = len(new_set & set(existing))
        if overlap >= min_overlap:
            return False
    return True


def select_diverse_ball_combinations(
    all_combos: List[BallCombination],
    max_count: int = 100,
    min_overlap: int = 4
) -> List[BallCombination]:
    """
    다양성 필터링으로 조합 선택
    - 이상치 점수 낮은 순 정렬
    - min_overlap자리 이상 중복 제거 (기본 4)
    - 조합 수 부족 시 min_overlap을 5로 재시도
    """
    # 점수 낮은 순 정렬
    sorted_combos = sorted(all_combos, key=lambda x: x.score)

    selected = []
    selected_numbers = []

    for combo in sorted_combos:
        if is_diverse_ball(combo.numbers, selected_numbers, min_overlap):
            selected.append(combo)
            selected_numbers.append(combo.numbers)
            if len(selected) >= max_count:
                break

    # 조합 수 부족 시 더 관대한 필터링으로 재시도
    if len(selected) < max_count and min_overlap < 5:
        return select_diverse_ball_combinations(all_combos, max_count, min_overlap + 1)

    return selected


def format_ball_combination(numbers: Tuple[int, ...]) -> str:
    """ball 조합을 문자열로 포맷"""
    return f"({', '.join(f'{n:2d}' for n in numbers)})"


def compare_ball_with_actual(predictions: List[BallCombination], actual: Tuple[int, ...]) -> Dict:
    """
    예측 조합들을 실제 당첨번호와 비교 (집합 비교)
    """
    actual_set = set(actual)

    best_match_count = 0
    best_combo = None
    match_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for combo in predictions:
        match_count = len(set(combo.numbers) & actual_set)
        match_distribution[match_count] += 1

        if match_count > best_match_count:
            best_match_count = match_count
            best_combo = combo

    return {
        'best_match_count': best_match_count,
        'best_combo': best_combo,
        'is_perfect': best_match_count == 6,
        'distribution': match_distribution
    }
