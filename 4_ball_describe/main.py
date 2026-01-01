#!/usr/bin/env python3
"""
Ball 순서 기반 데이터 분석 모듈

ball1~ball6: 추첨 순서 (1번째~6번째로 나온 공)
ord1~ord6: 오름차순 정렬된 번호

winning_numbers.csv를 읽어 Ball 관점의 describe_ball.csv 생성
"""

import csv
from pathlib import Path
from typing import Dict, List
from collections import Counter

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
OUTPUT_PATH = Path(__file__).parent / "describe_ball.csv"

# 소수 집합
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data() -> List[Dict]:
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = {
                'round': int(row['round']),
            }
            # ball1~ball6 추가
            for i in range(1, 7):
                data[f'ball{i}'] = int(row[f'ball{i}'])
            # ord1~ord6 추가 (참고용)
            for i in range(1, 7):
                data[f'ord{i}'] = int(row[f'ord{i}'])
            results.append(data)
    return sorted(results, key=lambda x: x['round'])


def get_range_code(num: int) -> str:
    """번호를 구간 코드로 변환"""
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


def analyze_ball_range(row: Dict) -> Dict:
    """Ball 범위/구간 분석"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    result = {
        'ball_sum': sum(balls),
        'ball_avg': round(sum(balls) / 6, 2),
        'ball_range_code': ''.join(get_range_code(n) for n in balls),
    }

    # 각 Ball 위치별 구간
    for i, n in enumerate(balls, 1):
        result[f'ball{i}_range'] = get_range_code(n)

    return result


def analyze_ball_prime(row: Dict) -> Dict:
    """Ball 소수 분석"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    prime_count = sum(1 for n in balls if n in PRIMES)
    prime_positions = [f'ball{i+1}' for i, n in enumerate(balls) if n in PRIMES]

    return {
        'ball_prime_count': prime_count,
        'ball_prime_positions': ','.join(prime_positions) if prime_positions else '',
    }


def analyze_ball_oddeven(row: Dict) -> Dict:
    """Ball 홀짝 분석"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    odd_count = sum(1 for n in balls if n % 2 == 1)
    pattern = ''.join('O' if n % 2 == 1 else 'E' for n in balls)

    return {
        'ball_odd_count': odd_count,
        'ball_even_count': 6 - odd_count,
        'ball_oddeven_pattern': pattern,
    }


def analyze_ball_position_stats(row: Dict) -> Dict:
    """각 Ball 위치별 통계"""
    result = {}

    for i in range(1, 7):
        ball = row[f'ball{i}']
        result[f'ball{i}_is_prime'] = ball in PRIMES
        result[f'ball{i}_is_odd'] = ball % 2 == 1
        result[f'ball{i}_lastdigit'] = ball % 10

    return result


def analyze_ball_carryover(prev: Dict, curr: Dict) -> Dict:
    """Ball 이월 분석"""
    if prev is None:
        return {
            'ball_carryover_count': 0,
            'ball_carryover_numbers': '',
        }

    prev_balls = set(prev[f'ball{i}'] for i in range(1, 7))
    curr_balls = [curr[f'ball{i}'] for i in range(1, 7)]

    carryover = [b for b in curr_balls if b in prev_balls]

    return {
        'ball_carryover_count': len(carryover),
        'ball_carryover_numbers': ','.join(str(n) for n in carryover) if carryover else '',
    }


def analyze_ball_position_carryover(prev: Dict, curr: Dict) -> Dict:
    """Ball 위치별 이월 분석 (같은 ball 위치에 같은 번호)"""
    if prev is None:
        return {
            'ball_pos_carryover': 0,
            'ball_pos_carryover_positions': '',
        }

    same_pos = []
    for i in range(1, 7):
        if prev[f'ball{i}'] == curr[f'ball{i}']:
            same_pos.append(f'ball{i}')

    return {
        'ball_pos_carryover': len(same_pos),
        'ball_pos_carryover_positions': ','.join(same_pos) if same_pos else '',
    }


def analyze_ball_range_distribution(row: Dict) -> Dict:
    """Ball 구간별 분포"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    count = {
        'ball_count_1to9': 0,
        'ball_count_10to19': 0,
        'ball_count_20to29': 0,
        'ball_count_30to39': 0,
        'ball_count_40to45': 0,
    }

    for n in balls:
        if 1 <= n <= 9:
            count['ball_count_1to9'] += 1
        elif 10 <= n <= 19:
            count['ball_count_10to19'] += 1
        elif 20 <= n <= 29:
            count['ball_count_20to29'] += 1
        elif 30 <= n <= 39:
            count['ball_count_30to39'] += 1
        else:
            count['ball_count_40to45'] += 1

    return count


def analyze_ball_sequence(row: Dict) -> Dict:
    """Ball 시퀀스 분석 (연속 증가/감소)"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    # 증가 패턴
    increases = sum(1 for i in range(5) if balls[i+1] > balls[i])
    decreases = sum(1 for i in range(5) if balls[i+1] < balls[i])

    # 최대 연속 증가/감소
    max_inc = 0
    max_dec = 0
    curr_inc = 0
    curr_dec = 0

    for i in range(5):
        if balls[i+1] > balls[i]:
            curr_inc += 1
            curr_dec = 0
            max_inc = max(max_inc, curr_inc)
        elif balls[i+1] < balls[i]:
            curr_dec += 1
            curr_inc = 0
            max_dec = max(max_dec, curr_dec)
        else:
            curr_inc = 0
            curr_dec = 0

    return {
        'ball_increase_count': increases,
        'ball_decrease_count': decreases,
        'ball_max_increase_seq': max_inc,
        'ball_max_decrease_seq': max_dec,
    }


def analyze_ball_gaps(row: Dict) -> Dict:
    """Ball 간 차이 분석"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    gaps = [abs(balls[i+1] - balls[i]) for i in range(5)]

    return {
        'ball_gap_12': gaps[0],
        'ball_gap_23': gaps[1],
        'ball_gap_34': gaps[2],
        'ball_gap_45': gaps[3],
        'ball_gap_56': gaps[4],
        'ball_gap_avg': round(sum(gaps) / 5, 2),
        'ball_gap_max': max(gaps),
        'ball_gap_min': min(gaps),
    }


def analyze_ball_first_last(row: Dict) -> Dict:
    """Ball1, Ball6 분석"""
    ball1 = row['ball1']
    ball6 = row['ball6']

    return {
        'ball1_range': get_range_code(ball1),
        'ball6_range': get_range_code(ball6),
        'ball1_is_prime': ball1 in PRIMES,
        'ball6_is_prime': ball6 in PRIMES,
        'ball1_ball6_diff': abs(ball6 - ball1),
        'ball1_is_min': ball1 == min(row[f'ball{i}'] for i in range(1, 7)),
        'ball6_is_max': ball6 == max(row[f'ball{i}'] for i in range(1, 7)),
    }


def analyze_ball_vs_ord(row: Dict) -> Dict:
    """Ball과 Ord 비교 분석"""
    balls = tuple(row[f'ball{i}'] for i in range(1, 7))
    ords = tuple(row[f'ord{i}'] for i in range(1, 7))

    # Ball1이 Ord 어디에 해당하는지
    ball1_ord_pos = ords.index(balls[0]) + 1 if balls[0] in ords else 0
    ball6_ord_pos = ords.index(balls[5]) + 1 if balls[5] in ords else 0

    return {
        'ball1_ord_position': ball1_ord_pos,
        'ball6_ord_position': ball6_ord_pos,
        'ball_is_ascending': balls == tuple(sorted(balls)),
        'ball_is_descending': balls == tuple(sorted(balls, reverse=True)),
    }


def analyze_ball_lastdigit(row: Dict) -> Dict:
    """Ball 끝수 분석"""
    balls = [row[f'ball{i}'] for i in range(1, 7)]

    last_digits = [n % 10 for n in balls]
    unique_digits = len(set(last_digits))

    # 연속 동일 끝수
    same_digit_streak = 0
    curr_streak = 1
    for i in range(1, 6):
        if last_digits[i] == last_digits[i-1]:
            curr_streak += 1
            same_digit_streak = max(same_digit_streak, curr_streak)
        else:
            curr_streak = 1

    return {
        'ball_lastdigit_pattern': ''.join(str(d) for d in last_digits),
        'ball_lastdigit_unique': unique_digits,
        'ball_lastdigit_streak': same_digit_streak,
    }


def get_ball_segments(all_data: List[Dict], target_round: int) -> Dict:
    """최근 10회차 기준 Ball 세그먼트 계산"""
    recent = [d for d in all_data if d['round'] < target_round][-10:]
    if len(recent) < 10:
        return None

    # 각 ball 위치별 빈도
    segments = {}
    for pos in range(1, 7):
        freq = Counter(d[f'ball{pos}'] for d in recent)
        sorted_nums = sorted(range(1, 46), key=lambda n: -freq.get(n, 0))
        segments[f'ball{pos}_top10'] = set(sorted_nums[:10])
        segments[f'ball{pos}_mid20'] = set(sorted_nums[10:30])
        segments[f'ball{pos}_rest15'] = set(sorted_nums[30:])

    return segments


def analyze_ball_segment(row: Dict, segments: Dict) -> Dict:
    """Ball 세그먼트 분석"""
    if segments is None:
        return {
            'ball_segment_code': '',
            'ball_in_top10': 0,
        }

    code = []
    in_top10 = 0

    for pos in range(1, 7):
        ball = row[f'ball{pos}']
        if ball in segments[f'ball{pos}_top10']:
            code.append('T')
            in_top10 += 1
        elif ball in segments[f'ball{pos}_mid20']:
            code.append('M')
        else:
            code.append('R')

    return {
        'ball_segment_code': ''.join(code),
        'ball_in_top10': in_top10,
    }


def generate_ball_insights(data: List[Dict]) -> Dict:
    """Ball 인사이트 통계 생성"""
    insights = {}
    n = len(data)

    # 1. Ball1 구간 분포
    ball1_ranges = Counter(d.get('ball1_range', 'A') for d in data)
    for r in ['A', 'B', 'C', 'D', 'E']:
        insights[f'ins_ball1_range_{r}'] = round(ball1_ranges.get(r, 0) / n * 100, 1)

    # 2. Ball6 구간 분포
    ball6_ranges = Counter(d.get('ball6_range', 'E') for d in data)
    for r in ['A', 'B', 'C', 'D', 'E']:
        insights[f'ins_ball6_range_{r}'] = round(ball6_ranges.get(r, 0) / n * 100, 1)

    # 3. Ball 이월수 분포
    carry_counts = Counter(d.get('ball_carryover_count', 0) for d in data)
    for k in range(5):
        insights[f'ins_ball_carryover_{k}'] = round(carry_counts.get(k, 0) / n * 100, 1)

    # 4. Ball 위치 이월 분포
    pos_carry = Counter(d.get('ball_pos_carryover', 0) for d in data)
    for k in range(4):
        insights[f'ins_ball_pos_carryover_{k}'] = round(pos_carry.get(k, 0) / n * 100, 1)

    # 5. Ball1이 최소인 비율
    ball1_min = sum(1 for d in data if d.get('ball1_is_min', False))
    insights['ins_ball1_is_min'] = round(ball1_min / n * 100, 1)

    # 6. Ball6이 최대인 비율
    ball6_max = sum(1 for d in data if d.get('ball6_is_max', False))
    insights['ins_ball6_is_max'] = round(ball6_max / n * 100, 1)

    # 7. 증가 카운트 분포
    inc_counts = Counter(d.get('ball_increase_count', 2) for d in data)
    for k in range(6):
        insights[f'ins_ball_increase_{k}'] = round(inc_counts.get(k, 0) / n * 100, 1)

    # 8. Ball 세그먼트 분포
    seg_data = [d for d in data if d.get('ball_segment_code', '')]
    if seg_data:
        seg_n = len(seg_data)
        top10_counts = Counter(d['ball_in_top10'] for d in seg_data)
        for k in range(7):
            insights[f'ins_ball_in_top10_{k}'] = round(top10_counts.get(k, 0) / seg_n * 100, 1)

    return insights


def add_ball_insight_flags(row: Dict, insights: Dict) -> Dict:
    """Ball 인사이트 기반 플래그 추가"""
    flags = {}
    outliers = []

    # Ball 이월수: 0~1개가 일반적
    carryover = row.get('ball_carryover_count', 0)
    if carryover <= 1:
        flags['ball_flag_carryover_normal'] = 1
    else:
        flags['ball_flag_carryover_normal'] = 0
        outliers.append(f"Ball이월:{carryover}개")

    # Ball 위치 이월: 0~1개가 일반적
    pos_carry = row.get('ball_pos_carryover', 0)
    if pos_carry <= 1:
        flags['ball_flag_pos_carryover_normal'] = 1
    else:
        flags['ball_flag_pos_carryover_normal'] = 0
        outliers.append(f"위치이월:{pos_carry}개")

    # Ball in Top10: 1~4개가 일반적
    in_top10 = row.get('ball_in_top10', 2)
    if 1 <= in_top10 <= 4:
        flags['ball_flag_top10_normal'] = 1
    else:
        flags['ball_flag_top10_normal'] = 0
        outliers.append(f"Top10:{in_top10}개")

    # 정상 점수
    flags['ball_normal_score'] = sum(v for k, v in flags.items() if k.startswith('ball_flag_'))
    flags['ball_outlier_count'] = len(outliers)
    flags['ball_outlier_list'] = ', '.join(outliers) if outliers else ''

    return flags


def analyze_ball_row(prev: Dict, curr: Dict, all_data: List[Dict]) -> Dict:
    """한 회차의 모든 Ball 분석 수행"""
    result = {
        'round': curr['round'],
    }

    # Ball 번호 추가
    for i in range(1, 7):
        result[f'ball{i}'] = curr[f'ball{i}']

    # Ord 번호 추가 (참고용)
    for i in range(1, 7):
        result[f'ord{i}'] = curr[f'ord{i}']

    # 모든 분석 함수 호출
    result.update(analyze_ball_range(curr))
    result.update(analyze_ball_prime(curr))
    result.update(analyze_ball_oddeven(curr))
    result.update(analyze_ball_position_stats(curr))
    result.update(analyze_ball_carryover(prev, curr))
    result.update(analyze_ball_position_carryover(prev, curr))
    result.update(analyze_ball_range_distribution(curr))
    result.update(analyze_ball_sequence(curr))
    result.update(analyze_ball_gaps(curr))
    result.update(analyze_ball_first_last(curr))
    result.update(analyze_ball_vs_ord(curr))
    result.update(analyze_ball_lastdigit(curr))

    # Ball 세그먼트 분석
    segments = get_ball_segments(all_data, curr['round'])
    result.update(analyze_ball_segment(curr, segments))

    return result


def save_ball_insights_csv(insights: Dict, output_path: Path):
    """Ball 인사이트 통계를 별도 CSV로 저장"""
    insights_path = output_path.parent / "insights_ball.csv"

    rows = []
    for key, value in insights.items():
        rows.append({'metric': key, 'value': value})

    with open(insights_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ball 인사이트 저장: {insights_path}")


def main():
    """전체 Ball 분석 및 CSV 저장"""
    print("Ball 데이터 로드 중...")
    data = load_data()
    print(f"총 {len(data)}개 회차 로드 완료")

    print("1단계: Ball 기본 분석 중...")
    results = []

    for i, row in enumerate(data):
        prev = data[i-1] if i > 0 else None
        result = analyze_ball_row(prev, row, data)
        results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  진행: {i + 1}/{len(data)}")

    print("2단계: Ball 인사이트 통계 생성 중...")
    insights = generate_ball_insights(results)

    print("3단계: Ball 정상 범위 플래그 추가 중...")
    for result in results:
        flags = add_ball_insight_flags(result, insights)
        result.update(flags)

    # CSV 저장
    print(f"저장 중: {OUTPUT_PATH}")

    fieldnames = list(results[0].keys())

    with open(OUTPUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 인사이트 CSV 저장
    save_ball_insights_csv(insights, OUTPUT_PATH)

    print(f"\n완료!")
    print(f"  - describe_ball.csv: {len(results)}개 회차, {len(fieldnames)}개 컬럼")
    print(f"  - insights_ball.csv: {len(insights)}개 통계")

    print(f"\nBall 인사이트 요약:")
    print(f"  Ball1 구간 A(1-9): {insights.get('ins_ball1_range_A', 0):.1f}%")
    print(f"  Ball6 구간 E(40-45): {insights.get('ins_ball6_range_E', 0):.1f}%")
    print(f"  Ball1이 최소: {insights.get('ins_ball1_is_min', 0):.1f}%")
    print(f"  Ball6이 최대: {insights.get('ins_ball6_is_max', 0):.1f}%")
    print(f"  Ball 이월 0개: {insights.get('ins_ball_carryover_0', 0):.1f}%")


if __name__ == '__main__':
    main()
