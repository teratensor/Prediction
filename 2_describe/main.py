#!/usr/bin/env python3
"""
회차별 데이터 분석 모듈

winning_numbers.csv를 읽어 각 회차의 모든 특징을 분석한 describe.csv 생성
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
OUTPUT_PATH = Path(__file__).parent / "describe.csv"

# 소수 집합
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data() -> List[Dict]:
    """당첨번호 데이터 로드 (o1~o45, ball1~ball6 포함)"""
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
            # o1~o45 추가 (shortcode 분석용)
            for i in range(1, 46):
                data[f'o{i}'] = int(row[f'o{i}'])
            # ball1~ball6 추가 (Ball 세그먼트 계산용)
            for i in range(1, 7):
                data[f'ball{i}'] = int(row[f'ball{i}'])
            results.append(data)
    return sorted(results, key=lambda x: x['round'])


def get_range_code(num: int) -> str:
    """번호를 구간 코드로 변환 (A=1-9, B=10-19, C=20-29, D=30-39, E=40-45)"""
    if 1 <= num <= 9:
        return 'A'
    elif 10 <= num <= 19:
        return 'B'
    elif 20 <= num <= 29:
        return 'C'
    elif 30 <= num <= 39:
        return 'D'
    else:
        return 'E'


def analyze_range(row: Dict) -> Dict:
    """범위/구간 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    result = {
        'span': row['ord6'] - row['ord1'],
        'sum6': sum(nums),
        'avg6': round(sum(nums) / 6, 2),
        'range_code': ''.join(get_range_code(n) for n in nums),
    }

    # 각 위치별 구간
    for i, n in enumerate(nums, 1):
        result[f'ord{i}_range'] = get_range_code(n)

    return result


def analyze_prime(row: Dict) -> Dict:
    """소수 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    prime_positions = []
    prime_numbers = []

    for i, n in enumerate(nums, 1):
        if n in PRIMES:
            prime_positions.append(f'ord{i}')
            prime_numbers.append(str(n))

    return {
        'prime_count': len(prime_numbers),
        'prime_positions': ','.join(prime_positions) if prime_positions else '',
        'prime_numbers': ','.join(prime_numbers) if prime_numbers else '',
    }


def analyze_oddeven(row: Dict) -> Dict:
    """홀짝 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    odd_count = sum(1 for n in nums if n % 2 == 1)
    pattern = ''.join('O' if n % 2 == 1 else 'E' for n in nums)

    return {
        'odd_count': odd_count,
        'even_count': 6 - odd_count,
        'oddeven_pattern': pattern,
    }


def analyze_consecutive(row: Dict) -> Dict:
    """연속수 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    groups = []
    current_group = [nums[0]]
    positions = []
    current_positions = ['ord1']

    for i in range(1, 6):
        if nums[i] == nums[i-1] + 1:
            current_group.append(nums[i])
            current_positions.append(f'ord{i+1}')
        else:
            if len(current_group) >= 2:
                groups.append(tuple(current_group))
                positions.append(','.join(current_positions))
            current_group = [nums[i]]
            current_positions = [f'ord{i+1}']

    # 마지막 그룹 처리
    if len(current_group) >= 2:
        groups.append(tuple(current_group))
        positions.append(','.join(current_positions))

    max_len = max((len(g) for g in groups), default=0)

    return {
        'consecutive_count': len(groups),
        'consecutive_max_len': max_len,
        'consecutive_groups': ';'.join(str(g) for g in groups) if groups else '',
        'consecutive_positions': ';'.join(positions) if positions else '',
    }


def analyze_gaps(row: Dict) -> Dict:
    """간격 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    gaps = [nums[i+1] - nums[i] for i in range(5)]

    def gap_code(g: int) -> str:
        if g <= 2:
            return 'S'  # Small
        elif g <= 5:
            return 'M'  # Medium
        elif g <= 10:
            return 'L'  # Large
        else:
            return 'X'  # eXtra large

    return {
        'gap_12': gaps[0],
        'gap_23': gaps[1],
        'gap_34': gaps[2],
        'gap_45': gaps[3],
        'gap_56': gaps[4],
        'gap_pattern': ''.join(gap_code(g) for g in gaps),
    }


def analyze_carryover(prev: Dict, curr: Dict) -> Dict:
    """이월수 분석 (전회차 대비)"""
    if prev is None:
        return {
            'carryover_count': 0,
            'carryover_numbers': '',
            'carryover_positions_prev': '',
            'carryover_positions_curr': '',
        }

    prev_nums = {prev[f'ord{i}']: f'ord{i}' for i in range(1, 7)}
    curr_nums = {curr[f'ord{i}']: f'ord{i}' for i in range(1, 7)}

    carryover = []
    prev_pos = []
    curr_pos = []

    for num, pos in curr_nums.items():
        if num in prev_nums:
            carryover.append(str(num))
            prev_pos.append(prev_nums[num])
            curr_pos.append(pos)

    return {
        'carryover_count': len(carryover),
        'carryover_numbers': ','.join(carryover) if carryover else '',
        'carryover_positions_prev': ','.join(prev_pos) if prev_pos else '',
        'carryover_positions_curr': ','.join(curr_pos) if curr_pos else '',
    }


def analyze_ac(row: Dict) -> Dict:
    """AC값 (Arithmetic Complexity) 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # 모든 쌍의 차이값 계산
    diffs = set()
    for i in range(6):
        for j in range(i+1, 6):
            diffs.add(nums[j] - nums[i])

    # AC = 고유 차이값 수 - 5
    ac_value = len(diffs) - 5

    return {
        'ac_value': ac_value,
    }


def analyze_last_digit(row: Dict) -> Dict:
    """끝수 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    last_digits = [n % 10 for n in nums]
    pattern = ''.join(str(d) for d in last_digits)
    unique_count = len(set(last_digits))

    # 동일 끝수 쌍 찾기
    digit_groups = {}
    for n in nums:
        d = n % 10
        if d not in digit_groups:
            digit_groups[d] = []
        digit_groups[d].append(n)

    same_pairs = []
    for d, group in digit_groups.items():
        if len(group) >= 2:
            same_pairs.extend(group)

    pair_count = sum(1 for g in digit_groups.values() if len(g) >= 2)

    return {
        'last_digit_pattern': pattern,
        'last_digit_unique': unique_count,
        'same_lastdigit_pairs': pair_count,
        'same_lastdigit_numbers': ','.join(str(n) for n in sorted(same_pairs)) if same_pairs else '',
    }


def analyze_range_distribution(row: Dict) -> Dict:
    """구간별 분포 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    count = {
        'count_1to9': 0,
        'count_10to19': 0,
        'count_20to29': 0,
        'count_30to39': 0,
        'count_40to45': 0,
    }

    for n in nums:
        if 1 <= n <= 9:
            count['count_1to9'] += 1
        elif 10 <= n <= 19:
            count['count_10to19'] += 1
        elif 20 <= n <= 29:
            count['count_20to29'] += 1
        elif 30 <= n <= 39:
            count['count_30to39'] += 1
        else:
            count['count_40to45'] += 1

    return count


def analyze_bonus(row: Dict) -> Dict:
    """보너스 분석"""
    bonus = row['ord_bonus']
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # 가장 가까운 당첨번호와 거리
    min_gap = min(abs(bonus - n) for n in nums)

    return {
        'bonus_range': get_range_code(bonus),
        'bonus_is_prime': bonus in PRIMES,
        'bonus_gap_to_nearest': min_gap,
    }


def analyze_range_carryover(prev: Dict, curr: Dict) -> Dict:
    """구간 분포 이월 분석"""
    if prev is None:
        return {
            'range_carryover_count': 0,
            'range_carryover_detail': '',
        }

    # 전회차와 현회차의 구간별 개수 계산
    def get_range_counts(row):
        nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]
        counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        for n in nums:
            counts[get_range_code(n)] += 1
        return counts

    prev_counts = get_range_counts(prev)
    curr_counts = get_range_counts(curr)

    # 동일 구간 분포 찾기
    same_count = 0
    details = []
    for r in ['A', 'B', 'C', 'D', 'E']:
        if prev_counts[r] == curr_counts[r] and prev_counts[r] > 0:
            same_count += 1
            details.append(f"{r}:{prev_counts[r]}→{curr_counts[r]}")

    return {
        'range_carryover_count': same_count,
        'range_carryover_detail': ', '.join(details) if details else '',
    }


def analyze_bonus_carryover(prev: Dict, curr: Dict) -> Dict:
    """보너스 이월 분석 (전회차 보너스가 현회차 본번호에 포함되는지)"""
    if prev is None:
        return {
            'prev_bonus_in_curr': False,
        }

    prev_bonus = prev['ord_bonus']
    curr_nums = [curr['ord1'], curr['ord2'], curr['ord3'], curr['ord4'], curr['ord5'], curr['ord6']]

    return {
        'prev_bonus_in_curr': prev_bonus in curr_nums,
    }


def analyze_bonus_extended(row: Dict) -> Dict:
    """보너스 확장 분석"""
    bonus = row['ord_bonus']
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # 보너스 홀짝
    bonus_odd = 'O' if bonus % 2 == 1 else 'E'

    # 보너스 끝자리
    bonus_last_digit = bonus % 10

    # 보너스가 본번호 범위 내에 있는지
    bonus_in_main = nums[0] < bonus < nums[5]

    # 보너스가 정렬 시 들어갈 위치 (0~6, 0은 ord1보다 작음, 6은 ord6보다 큼)
    if bonus < nums[0]:
        bonus_position = 0
    elif bonus > nums[5]:
        bonus_position = 6
    else:
        for i in range(5):
            if nums[i] < bonus < nums[i+1]:
                bonus_position = i + 1
                break
        else:
            bonus_position = -1  # shouldn't happen

    return {
        'bonus_odd': bonus_odd,
        'bonus_last_digit': bonus_last_digit,
        'bonus_in_main': bonus_in_main,
        'bonus_position': bonus_position,
    }


def analyze_range_pattern(row: Dict) -> Dict:
    """구간 패턴 분석 (빈 구간, 최다 구간)"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for n in nums:
        counts[get_range_code(n)] += 1

    # 빈 구간
    empty = [r for r, c in counts.items() if c == 0]

    # 최다 구간
    max_count = max(counts.values())
    dominant = [r for r, c in counts.items() if c == max_count]

    return {
        'empty_ranges': ','.join(empty) if empty else '',
        'empty_range_count': len(empty),
        'dominant_range': ','.join(dominant),
        'dominant_range_count': max_count,
    }


def analyze_distribution(row: Dict) -> Dict:
    """번호 분포 추가 분석 (저/고번호, 중앙값)"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # 저번호 (1-22), 고번호 (23-45)
    low_count = sum(1 for n in nums if n <= 22)
    high_count = 6 - low_count

    # 중앙값
    median = (nums[2] + nums[3]) / 2

    return {
        'low_count': low_count,
        'high_count': high_count,
        'median': median,
    }


def analyze_lastdigit_carryover(prev: Dict, curr: Dict) -> Dict:
    """끝수 이월 분석"""
    if prev is None:
        return {
            'lastdigit_carryover_count': 0,
            'lastdigit_carryover_digits': '',
        }

    prev_nums = [prev['ord1'], prev['ord2'], prev['ord3'], prev['ord4'], prev['ord5'], prev['ord6']]
    curr_nums = [curr['ord1'], curr['ord2'], curr['ord3'], curr['ord4'], curr['ord5'], curr['ord6']]

    prev_digits = set(n % 10 for n in prev_nums)
    curr_digits = set(n % 10 for n in curr_nums)

    # 이월된 끝자리
    carryover_digits = sorted(prev_digits & curr_digits)

    return {
        'lastdigit_carryover_count': len(carryover_digits),
        'lastdigit_carryover_digits': ','.join(str(d) for d in carryover_digits) if carryover_digits else '',
    }


def analyze_statistics(row: Dict) -> Dict:
    """통계적 분석 (분산, 표준편차, IQR, 변동계수)"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # 기본 통계
    mean = sum(nums) / 6
    variance = sum((n - mean) ** 2 for n in nums) / 6
    std = variance ** 0.5

    # IQR (사분위수 범위)
    q1 = (nums[1] + nums[2]) / 2  # 25%
    q3 = (nums[3] + nums[4]) / 2  # 75%
    iqr = q3 - q1

    # 변동계수 (CV) - 상대적 분산
    cv = (std / mean) * 100 if mean > 0 else 0

    return {
        'stat_variance': round(variance, 2),
        'stat_std': round(std, 2),
        'stat_q1': q1,
        'stat_q3': q3,
        'stat_iqr': iqr,
        'stat_cv': round(cv, 2),
    }


def analyze_skewness(row: Dict) -> Dict:
    """왜도/분포 치우침 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    mean = sum(nums) / 6
    variance = sum((n - mean) ** 2 for n in nums) / 6
    std = variance ** 0.5

    # 왜도 (Skewness) - 음수=저번호 치우침, 양수=고번호 치우침
    if std > 0:
        skewness = sum(((n - mean) / std) ** 3 for n in nums) / 6
    else:
        skewness = 0

    # 왜도 분류
    if skewness < -0.3:
        skew_type = 'low'  # 저번호 치우침
    elif skewness > 0.3:
        skew_type = 'high'  # 고번호 치우침
    else:
        skew_type = 'balanced'  # 균형

    return {
        'stat_skewness': round(skewness, 3),
        'stat_skew_type': skew_type,
    }


def analyze_gap_uniformity(row: Dict) -> Dict:
    """간격 균등도 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]
    gaps = [nums[i+1] - nums[i] for i in range(5)]

    # 간격 통계
    gap_mean = sum(gaps) / 5
    gap_variance = sum((g - gap_mean) ** 2 for g in gaps) / 5
    gap_std = gap_variance ** 0.5

    # 균등도 지수 (0에 가까울수록 균등)
    # 이상적 균등 간격 = span / 5
    ideal_gap = (nums[5] - nums[0]) / 5
    uniformity = sum(abs(g - ideal_gap) for g in gaps) / 5

    # 균등도 분류
    if gap_std < 3:
        gap_type = 'uniform'  # 균등
    elif gap_std < 6:
        gap_type = 'moderate'  # 보통
    else:
        gap_type = 'varied'  # 불균등

    return {
        'stat_gap_std': round(gap_std, 2),
        'stat_gap_uniformity': round(uniformity, 2),
        'stat_gap_type': gap_type,
    }


def analyze_fibonacci(row: Dict) -> Dict:
    """피보나치 관련성 분석"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # 1-45 범위 내 피보나치 수
    fib_set = {1, 2, 3, 5, 8, 13, 21, 34}

    # 피보나치 수 개수
    fib_count = sum(1 for n in nums if n in fib_set)
    fib_numbers = [n for n in nums if n in fib_set]

    # 황금비 관련: 인접 번호 비율이 1.618에 가까운지
    golden_ratio = 1.618
    golden_pairs = 0
    for i in range(5):
        if nums[i] > 0:
            ratio = nums[i+1] / nums[i]
            if 1.4 < ratio < 1.9:  # 황금비 근사
                golden_pairs += 1

    return {
        'stat_fib_count': fib_count,
        'stat_fib_numbers': ','.join(str(n) for n in fib_numbers) if fib_numbers else '',
        'stat_golden_pairs': golden_pairs,
    }


def analyze_modular(row: Dict) -> Dict:
    """모듈러 패턴 분석 (3, 5, 7로 나눈 나머지 분포)"""
    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # mod 3 분포 (0, 1, 2)
    mod3 = [n % 3 for n in nums]
    mod3_counts = [mod3.count(i) for i in range(3)]

    # mod 5 분포 (0, 1, 2, 3, 4)
    mod5 = [n % 5 for n in nums]
    mod5_unique = len(set(mod5))

    # 완전 분포 여부 (mod5에서 5개 이상 다른 나머지)
    mod5_complete = mod5_unique >= 5

    return {
        'stat_mod3_0': mod3_counts[0],
        'stat_mod3_1': mod3_counts[1],
        'stat_mod3_2': mod3_counts[2],
        'stat_mod5_unique': mod5_unique,
        'stat_mod5_complete': mod5_complete,
    }


def get_segments(all_data: List[Dict], target_round: int) -> Dict:
    """최근 10회차 기준 Ord/Ball 세그먼트 계산 (Top24/Mid14/Rest7)"""
    recent = [d for d in all_data if d['round'] < target_round][-10:]
    if len(recent) < 10:
        return None  # 데이터 부족

    # Ord 세그먼트: ord1~ord6 기준 빈도
    ord_freq = Counter()
    for r in recent:
        for i in range(1, 7):
            ord_freq[r[f'ord{i}']] += 1

    sorted_ord = sorted(range(1, 46), key=lambda n: -ord_freq.get(n, 0))
    ord_top24 = set(sorted_ord[:24])
    ord_mid14 = set(sorted_ord[24:38])
    ord_rest7 = set(sorted_ord[38:])

    # Ball 세그먼트: ball1~ball6 기준 빈도
    ball_freq = Counter()
    for r in recent:
        for i in range(1, 7):
            ball_freq[r[f'ball{i}']] += 1

    sorted_ball = sorted(range(1, 46), key=lambda n: -ball_freq.get(n, 0))
    ball_top24 = set(sorted_ball[:24])
    ball_mid14 = set(sorted_ball[24:38])
    ball_rest7 = set(sorted_ball[38:])

    return {
        'ord_top24': ord_top24, 'ord_mid14': ord_mid14, 'ord_rest7': ord_rest7,
        'ball_top24': ball_top24, 'ball_mid14': ball_mid14, 'ball_rest7': ball_rest7,
    }


def analyze_shortcode(row: Dict, segments: Dict) -> Dict:
    """Shortcode 분석 (Ord + Ball 코드)

    - Ord 코드: 당첨번호(ord)가 Ord 세그먼트(ord 빈도 기준)에서 어디에 속하는지
    - Ball 코드: 당첨번호(ord)가 Ball 세그먼트(ball 빈도 기준)에서 어디에 속하는지
    """
    if segments is None:
        return {
            'shortcode': '',
            'ord_code': '',
            'ball_code': '',
            'sc_top24': 0,
            'sc_mid14': 0,
            'sc_rest7': 0,
            'ball_top24': 0,
            'ball_mid14': 0,
            'ball_rest7': 0,
            'sc_ord_eq_ball': False,
            'sc_pattern': '',
        }

    nums = [row['ord1'], row['ord2'], row['ord3'], row['ord4'], row['ord5'], row['ord6']]

    # Ord 코드: 당첨번호가 Ord 세그먼트에서 어디에 속하는지
    ord_top = sum(1 for n in nums if n in segments['ord_top24'])
    ord_mid = sum(1 for n in nums if n in segments['ord_mid14'])
    ord_rest = sum(1 for n in nums if n in segments['ord_rest7'])
    ord_code = f"{ord_top}{ord_mid}{ord_rest}"

    # Ball 코드: 당첨번호가 Ball 세그먼트에서 어디에 속하는지
    ball_top = sum(1 for n in nums if n in segments['ball_top24'])
    ball_mid = sum(1 for n in nums if n in segments['ball_mid14'])
    ball_rest = sum(1 for n in nums if n in segments['ball_rest7'])
    ball_code = f"{ball_top}{ball_mid}{ball_rest}"

    shortcode = ord_code + ball_code

    # 패턴 분류
    if ord_top >= 4:
        pattern = 'top_heavy'
    elif ord_top <= 2 and ord_rest >= 2:
        pattern = 'spread'
    else:
        pattern = 'balanced'

    return {
        'shortcode': shortcode,
        'ord_code': ord_code,
        'ball_code': ball_code,
        'sc_top24': ord_top,
        'sc_mid14': ord_mid,
        'sc_rest7': ord_rest,
        'ball_top24': ball_top,
        'ball_mid14': ball_mid,
        'ball_rest7': ball_rest,
        'sc_ord_eq_ball': ord_code == ball_code,
        'sc_pattern': pattern,
    }


def generate_insights(data: List[Dict]) -> Dict:
    """전체 데이터에서 인사이트 통계 생성"""
    insights = {}
    n = len(data)

    # 1. 연속수 분포
    consec_counts = Counter(d['consecutive_count'] for d in data)
    for k in range(4):
        insights[f'ins_consecutive_{k}'] = round(consec_counts.get(k, 0) / n * 100, 1)

    # 2. 소수 개수 분포
    prime_counts = Counter(d['prime_count'] for d in data)
    for k in range(7):
        insights[f'ins_prime_{k}'] = round(prime_counts.get(k, 0) / n * 100, 1)

    # 3. 홀수 개수 분포
    odd_counts = Counter(d['odd_count'] for d in data)
    for k in range(7):
        insights[f'ins_odd_{k}'] = round(odd_counts.get(k, 0) / n * 100, 1)

    # 4. 이월수 분포
    carry_counts = Counter(d['carryover_count'] for d in data)
    for k in range(5):
        insights[f'ins_carryover_{k}'] = round(carry_counts.get(k, 0) / n * 100, 1)

    # 5. AC값 분포
    ac_counts = Counter(d['ac_value'] for d in data)
    for k in range(3, 11):
        insights[f'ins_ac_{k}'] = round(ac_counts.get(k, 0) / n * 100, 1)

    # 6. 합계 구간 분포
    sum_ranges = {'70_99': 0, '100_119': 0, '120_139': 0, '140_159': 0, '160_179': 0, '180_plus': 0}
    for d in data:
        s = d['sum6']
        if s < 100:
            sum_ranges['70_99'] += 1
        elif s < 120:
            sum_ranges['100_119'] += 1
        elif s < 140:
            sum_ranges['120_139'] += 1
        elif s < 160:
            sum_ranges['140_159'] += 1
        elif s < 180:
            sum_ranges['160_179'] += 1
        else:
            sum_ranges['180_plus'] += 1
    for k, v in sum_ranges.items():
        insights[f'ins_sum_{k}'] = round(v / n * 100, 1)

    # 7. 동일 끝수 쌍 분포
    same_pair_counts = Counter(d['same_lastdigit_pairs'] for d in data)
    for k in range(4):
        insights[f'ins_samedigit_{k}'] = round(same_pair_counts.get(k, 0) / n * 100, 1)

    # 8. 범위(span) 통계
    spans = [d['span'] for d in data]
    insights['ins_span_min'] = min(spans)
    insights['ins_span_max'] = max(spans)
    insights['ins_span_avg'] = round(sum(spans) / n, 1)

    # 9. 합계 통계
    sums = [d['sum6'] for d in data]
    insights['ins_sum_min'] = min(sums)
    insights['ins_sum_max'] = max(sums)
    insights['ins_sum_avg'] = round(sum(sums) / n, 1)

    # 10. 왜도(skewness) 분포
    skew_types = Counter(d.get('stat_skew_type', 'balanced') for d in data)
    for k in ['low', 'balanced', 'high']:
        insights[f'ins_skew_{k}'] = round(skew_types.get(k, 0) / n * 100, 1)

    # 11. 간격 균등도 분포
    gap_types = Counter(d.get('stat_gap_type', 'moderate') for d in data)
    for k in ['uniform', 'moderate', 'varied']:
        insights[f'ins_gaptype_{k}'] = round(gap_types.get(k, 0) / n * 100, 1)

    # 12. 피보나치 개수 분포
    fib_counts = Counter(d.get('stat_fib_count', 0) for d in data)
    for k in range(5):
        insights[f'ins_fib_{k}'] = round(fib_counts.get(k, 0) / n * 100, 1)

    # 13. 황금비 쌍 분포
    golden_counts = Counter(d.get('stat_golden_pairs', 0) for d in data)
    for k in range(5):
        insights[f'ins_golden_{k}'] = round(golden_counts.get(k, 0) / n * 100, 1)

    # 14. 표준편차 구간 분포
    std_ranges = {'low': 0, 'normal': 0, 'high': 0}
    for d in data:
        std = d.get('stat_std', 12)
        if std < 8:
            std_ranges['low'] += 1
        elif std <= 14:
            std_ranges['normal'] += 1
        else:
            std_ranges['high'] += 1
    for k, v in std_ranges.items():
        insights[f'ins_std_{k}'] = round(v / n * 100, 1)

    # 15. mod5 완전성 분포
    mod5_complete = sum(1 for d in data if d.get('stat_mod5_complete', False))
    insights['ins_mod5_complete'] = round(mod5_complete / n * 100, 1)
    insights['ins_mod5_incomplete'] = round((n - mod5_complete) / n * 100, 1)

    # 16. 변동계수(CV) 구간 분포
    cv_ranges = {'low': 0, 'normal': 0, 'high': 0}
    for d in data:
        cv = d.get('stat_cv', 50)
        if cv < 40:
            cv_ranges['low'] += 1
        elif cv <= 60:
            cv_ranges['normal'] += 1
        else:
            cv_ranges['high'] += 1
    for k, v in cv_ranges.items():
        insights[f'ins_cv_{k}'] = round(v / n * 100, 1)

    # === Shortcode 인사이트 (첫 10회차 제외) ===
    sc_data = [d for d in data if d.get('shortcode', '')]
    sc_n = len(sc_data) if sc_data else 1  # 0으로 나누기 방지

    # 17. ord_code 분포 (상위 5개)
    ord_codes = Counter(d['ord_code'] for d in sc_data)
    for i, (code, cnt) in enumerate(ord_codes.most_common(5), 1):
        insights[f'ins_ordcode_{i}_{code}'] = round(cnt / sc_n * 100, 1)

    # 18. sc_pattern 분포
    patterns = Counter(d['sc_pattern'] for d in sc_data)
    for pat in ['balanced', 'top_heavy', 'spread']:
        insights[f'ins_scpattern_{pat}'] = round(patterns.get(pat, 0) / sc_n * 100, 1)

    # 19. sc_top24 분포
    top24_counts = Counter(d['sc_top24'] for d in sc_data)
    for k in range(7):
        insights[f'ins_sctop24_{k}'] = round(top24_counts.get(k, 0) / sc_n * 100, 1)

    # 20. sc_rest7 분포
    rest7_counts = Counter(d['sc_rest7'] for d in sc_data)
    for k in range(4):
        insights[f'ins_screst7_{k}'] = round(rest7_counts.get(k, 0) / sc_n * 100, 1)

    # 21. ball_code 분포 (상위 5개)
    ball_codes = Counter(d['ball_code'] for d in sc_data)
    for i, (code, cnt) in enumerate(ball_codes.most_common(5), 1):
        insights[f'ins_ballcode_{i}_{code}'] = round(cnt / sc_n * 100, 1)

    # 22. ball_top24 분포
    ball_top24_counts = Counter(d['ball_top24'] for d in sc_data)
    for k in range(7):
        insights[f'ins_balltop24_{k}'] = round(ball_top24_counts.get(k, 0) / sc_n * 100, 1)

    # 23. ball_rest7 분포
    ball_rest7_counts = Counter(d['ball_rest7'] for d in sc_data)
    for k in range(5):
        insights[f'ins_ballrest7_{k}'] = round(ball_rest7_counts.get(k, 0) / sc_n * 100, 1)

    # 24. ord_code와 ball_code 일치율
    eq_count = sum(1 for d in sc_data if d.get('sc_ord_eq_ball', False))
    insights['ins_sc_ord_eq_ball'] = round(eq_count / sc_n * 100, 1)

    return insights


def add_insight_flags(row: Dict, insights: Dict) -> Dict:
    """각 행에 인사이트 기반 플래그 추가 (정상 범위 내인지) + 이상치 기록"""
    flags = {}
    outliers = []

    # 연속수: 0~1개가 91%
    if row['consecutive_count'] <= 1:
        flags['flag_consecutive_normal'] = 1
    else:
        flags['flag_consecutive_normal'] = 0
        outliers.append(f"연속수:{row['consecutive_count']}개")

    # 소수: 1~3개가 83.6%
    if 1 <= row['prime_count'] <= 3:
        flags['flag_prime_normal'] = 1
    else:
        flags['flag_prime_normal'] = 0
        outliers.append(f"소수:{row['prime_count']}개")

    # 홀수: 2~4개가 84.1%
    if 2 <= row['odd_count'] <= 4:
        flags['flag_odd_normal'] = 1
    else:
        flags['flag_odd_normal'] = 0
        outliers.append(f"홀수:{row['odd_count']}개")

    # 이월수: 0~1개가 80.5%
    if row['carryover_count'] <= 1:
        flags['flag_carryover_normal'] = 1
    else:
        flags['flag_carryover_normal'] = 0
        outliers.append(f"이월:{row['carryover_count']}개")

    # AC값: 7~10이 83.1%
    if 7 <= row['ac_value'] <= 10:
        flags['flag_ac_normal'] = 1
    else:
        flags['flag_ac_normal'] = 0
        outliers.append(f"AC:{row['ac_value']}")

    # 합계: 100~159가 69.9%
    if 100 <= row['sum6'] <= 159:
        flags['flag_sum_normal'] = 1
    else:
        flags['flag_sum_normal'] = 0
        outliers.append(f"합계:{row['sum6']}")

    # 동일 끝수: 0~1쌍이 79.1%
    if row['same_lastdigit_pairs'] <= 1:
        flags['flag_samedigit_normal'] = 1
    else:
        flags['flag_samedigit_normal'] = 0
        outliers.append(f"동끝수:{row['same_lastdigit_pairs']}쌍")

    # === 새로운 통계적 이상치 탐지 ===

    # 왜도: balanced가 43%, low/high 각각 ~30%
    skew_type = row.get('stat_skew_type', 'balanced')
    if skew_type == 'balanced':
        flags['flag_skew_normal'] = 1
    else:
        flags['flag_skew_normal'] = 0
        skew_label = '저편중' if skew_type == 'low' else '고편중'
        outliers.append(f"왜도:{skew_label}")

    # 간격 균등도: uniform+moderate가 74.9%
    gap_type = row.get('stat_gap_type', 'moderate')
    if gap_type in ['uniform', 'moderate']:
        flags['flag_gaptype_normal'] = 1
    else:
        flags['flag_gaptype_normal'] = 0
        outliers.append(f"간격:불균등")

    # 표준편차: 8~14가 71.2%
    std_val = row.get('stat_std', 12)
    if 8 <= std_val <= 14:
        flags['flag_std_normal'] = 1
    else:
        flags['flag_std_normal'] = 0
        std_label = '집중' if std_val < 8 else '분산'
        outliers.append(f"std:{std_label}({std_val:.1f})")

    # 피보나치: 0~2개가 93.5%
    fib_count = row.get('stat_fib_count', 0)
    if fib_count <= 2:
        flags['flag_fib_normal'] = 1
    else:
        flags['flag_fib_normal'] = 0
        outliers.append(f"피보나치:{fib_count}개")

    # 황금비 쌍: 0~2쌍이 93.9%
    golden_pairs = row.get('stat_golden_pairs', 0)
    if golden_pairs <= 2:
        flags['flag_golden_normal'] = 1
    else:
        flags['flag_golden_normal'] = 0
        outliers.append(f"황금비:{golden_pairs}쌍")

    # === Shortcode 이상치 탐지 (첫 10회차는 데이터 없음) ===
    shortcode = row.get('shortcode', '')
    if shortcode:  # shortcode가 있는 경우만 검사
        # sc_top24: 2~4개가 79.9%
        sc_top24 = row.get('sc_top24', 3)
        if 2 <= sc_top24 <= 4:
            flags['flag_sctop24_normal'] = 1
        else:
            flags['flag_sctop24_normal'] = 0
            outliers.append(f"Top24:{sc_top24}개")

        # sc_rest7: 0~1개가 78.0%
        sc_rest7 = row.get('sc_rest7', 0)
        if sc_rest7 <= 1:
            flags['flag_screst7_normal'] = 1
        else:
            flags['flag_screst7_normal'] = 0
            outliers.append(f"Rest7:{sc_rest7}개")

        # sc_pattern: balanced/top_heavy가 90.5% (spread는 이상치)
        sc_pattern = row.get('sc_pattern', 'balanced')
        if sc_pattern in ['balanced', 'top_heavy']:
            flags['flag_scpattern_normal'] = 1
        else:
            flags['flag_scpattern_normal'] = 0
            outliers.append(f"SC패턴:spread")
    else:
        # 데이터 없는 경우 정상으로 처리
        flags['flag_sctop24_normal'] = 1
        flags['flag_screst7_normal'] = 1
        flags['flag_scpattern_normal'] = 1

    # 전체 정상 점수 (15개 중 몇 개 정상인지)
    flags['normal_score'] = sum(v for k, v in flags.items() if k.startswith('flag_'))

    # 이상치 기록
    flags['outlier_count'] = len(outliers)
    flags['outlier_list'] = ', '.join(outliers) if outliers else ''

    return flags


def analyze_row(prev: Dict, curr: Dict, all_data: List[Dict]) -> Dict:
    """한 회차의 모든 분석 수행"""
    result = {
        'round': curr['round'],
        'ord1': curr['ord1'],
        'ord2': curr['ord2'],
        'ord3': curr['ord3'],
        'ord4': curr['ord4'],
        'ord5': curr['ord5'],
        'ord6': curr['ord6'],
        'ord_bonus': curr['ord_bonus'],
    }

    # 모든 분석 함수 호출
    result.update(analyze_range(curr))
    result.update(analyze_prime(curr))
    result.update(analyze_oddeven(curr))
    result.update(analyze_consecutive(curr))
    result.update(analyze_gaps(curr))
    result.update(analyze_carryover(prev, curr))
    result.update(analyze_ac(curr))
    result.update(analyze_last_digit(curr))
    result.update(analyze_range_distribution(curr))
    result.update(analyze_bonus(curr))

    # 추가 분석 함수 호출
    result.update(analyze_range_carryover(prev, curr))
    result.update(analyze_bonus_carryover(prev, curr))
    result.update(analyze_bonus_extended(curr))
    result.update(analyze_range_pattern(curr))
    result.update(analyze_distribution(curr))
    result.update(analyze_lastdigit_carryover(prev, curr))

    # 통계적 분석 함수 호출
    result.update(analyze_statistics(curr))
    result.update(analyze_skewness(curr))
    result.update(analyze_gap_uniformity(curr))
    result.update(analyze_fibonacci(curr))
    result.update(analyze_modular(curr))

    # Shortcode 분석 (6_shortcode 인사이트)
    segments = get_segments(all_data, curr['round'])
    result.update(analyze_shortcode(curr, segments))

    return result


def save_insights_csv(insights: Dict, output_path: Path):
    """인사이트 통계를 별도 CSV로 저장"""
    insights_path = output_path.parent / "insights.csv"

    rows = []
    for key, value in insights.items():
        rows.append({'metric': key, 'value': value})

    with open(insights_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"인사이트 저장: {insights_path}")


def main():
    """전체 분석 및 CSV 저장"""
    print("데이터 로드 중...")
    data = load_data()
    print(f"총 {len(data)}개 회차 로드 완료")

    print("1단계: 기본 분석 중...")
    results = []

    for i, row in enumerate(data):
        prev = data[i-1] if i > 0 else None
        result = analyze_row(prev, row, data)
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  진행: {i + 1}/{len(data)}")

    print("2단계: 인사이트 통계 생성 중...")
    insights = generate_insights(results)

    print("3단계: 정상 범위 플래그 추가 중...")
    for result in results:
        flags = add_insight_flags(result, insights)
        result.update(flags)

    # CSV 저장
    print(f"저장 중: {OUTPUT_PATH}")

    fieldnames = list(results[0].keys())

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 인사이트 CSV 저장
    save_insights_csv(insights, OUTPUT_PATH)

    print(f"\n완료!")
    print(f"  - describe.csv: {len(results)}개 회차, {len(fieldnames)}개 컬럼")
    print(f"  - insights.csv: {len(insights)}개 통계")

    print(f"\n추가된 플래그 컬럼:")
    flag_cols = [c for c in fieldnames if c.startswith('flag_') or c == 'normal_score']
    for col in flag_cols:
        print(f"  - {col}")

    print(f"\n인사이트 요약:")
    print(f"  연속수 0~1개: {insights['ins_consecutive_0'] + insights['ins_consecutive_1']:.1f}%")
    print(f"  소수 1~3개: {insights['ins_prime_1'] + insights['ins_prime_2'] + insights['ins_prime_3']:.1f}%")
    print(f"  홀수 2~4개: {insights['ins_odd_2'] + insights['ins_odd_3'] + insights['ins_odd_4']:.1f}%")
    print(f"  이월수 0~1개: {insights['ins_carryover_0'] + insights['ins_carryover_1']:.1f}%")
    print(f"  AC값 7~10: {insights['ins_ac_7'] + insights['ins_ac_8'] + insights['ins_ac_9'] + insights['ins_ac_10']:.1f}%")
    print(f"  합계 100~159: {insights['ins_sum_100_119'] + insights['ins_sum_120_139'] + insights['ins_sum_140_159']:.1f}%")
    print(f"  동일끝수 0~1쌍: {insights['ins_samedigit_0'] + insights['ins_samedigit_1']:.1f}%")


if __name__ == '__main__':
    main()
