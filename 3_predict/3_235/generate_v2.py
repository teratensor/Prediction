"""
새로운 조합 생성기 v2

전략:
1. firstend 인사이트로 ord1(첫수), ord6(끝수) 쌍 선택
2. 모든 인사이트를 적용하여 45개 번호에 점수 부여
3. 점수순 정렬 후 상위 번호로 조합 생성
"""

import csv
import argparse
from pathlib import Path
from collections import Counter
from itertools import combinations

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent

# 소수 목록 (1~45)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data():
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
                'o_table': {i: int(row[f'o{i}']) for i in range(1, 46)}
            })
    return results


def get_recent_stats(data, n=50):
    """최근 N회차 통계"""
    recent = data[-n:]

    # 전체 빈도
    all_freq = Counter()
    for r in recent:
        for ball in r['balls']:
            all_freq[ball] += 1

    # 포지션별 빈도
    pos_freq = {i: Counter() for i in range(6)}
    for r in recent:
        for pos, ball in enumerate(r['balls']):
            pos_freq[pos][ball] += 1

    # 첫수(ball1) 빈도
    first_freq = Counter(r['balls'][0] for r in recent)

    # 끝수(ball6) 빈도
    last_freq = Counter(r['balls'][5] for r in recent)

    # (첫수, 끝수) 쌍 빈도
    pair_freq = Counter((r['balls'][0], r['balls'][5]) for r in recent)

    return {
        'all_freq': all_freq,
        'pos_freq': pos_freq,
        'first_freq': first_freq,
        'last_freq': last_freq,
        'pair_freq': pair_freq,
        'recent': recent
    }


def select_firstend_pairs(stats, top_ratio=0.75):
    """
    1단계: firstend 인사이트로 (첫수, 끝수) 쌍 선택

    점수 = first빈도 + last빈도 + pair빈도*2 - 최근3회패널티
    상위 75% 선택
    """
    first_freq = stats['first_freq']
    last_freq = stats['last_freq']
    pair_freq = stats['pair_freq']
    recent = stats['recent']

    # 최근 3회 출현 번호
    recent_3 = set()
    for r in recent[-3:]:
        recent_3.update(r['balls'])

    # 모든 가능한 (첫수, 끝수) 쌍 점수 계산
    pairs = []
    for first in range(1, 11):  # 첫수는 보통 1-10
        for last in range(first + 5, 46):  # 끝수는 첫수+5 이상
            score = 0
            score += first_freq.get(first, 0) * 2
            score += last_freq.get(last, 0) * 2
            score += pair_freq.get((first, last), 0) * 5

            # 최근 3회 출현 페널티
            if first in recent_3:
                score -= 3
            if last in recent_3:
                score -= 3

            pairs.append({
                'first': first,
                'last': last,
                'score': score
            })

    # 점수순 정렬
    pairs.sort(key=lambda x: -x['score'])

    # 상위 N% 선택
    n_select = int(len(pairs) * top_ratio)
    selected = pairs[:n_select]

    return selected


def score_number(num, stats, pos_hint=None):
    """
    2단계: 개별 번호에 대한 인사이트 점수 계산

    적용 인사이트:
    - 2_second: ball2 범위 (10-19가 48.3%)
    - 3_third: ball3 범위 (10-19가 45.4%)
    - 4_fourth: ball4 범위 (20-29가 42.7%)
    - 5_fifth: ball5 범위 (30-39가 54.6%)
    - 6_consecutive: 연속수 보너스/페널티
    - 7_lastnum: ball6 범위 (40-45가 57.8%)
    - 9_range: 구간 균형
    - 10_prime: 소수 보너스
    - 11_shortcode: Top24 빈도
    - 12_onehot: 핫/콜드 비트
    """
    all_freq = stats['all_freq']
    score = 0

    # 기본 빈도 점수 (최근 50회)
    score += all_freq.get(num, 0) * 2

    # 포지션별 빈도 (힌트가 있으면)
    if pos_hint is not None:
        pos_freq = stats['pos_freq'][pos_hint]
        score += pos_freq.get(num, 0) * 3

    # 소수 보너스 (1-2개가 66.7%)
    if num in PRIMES:
        score += 5

    # 구간별 보너스
    if 10 <= num <= 19:
        score += 3  # ball2, ball3 최빈 구간
    elif 20 <= num <= 29:
        score += 3  # ball4 최빈 구간
    elif 30 <= num <= 39:
        score += 4  # ball5 최빈 구간
    elif 40 <= num <= 45:
        score += 3  # ball6 최빈 구간

    # 핫 비트 보너스 (12_onehot)
    hot_bits = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
    cold_bits = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    if num in hot_bits:
        score += 4
    if num in cold_bits:
        score -= 2

    return score


def score_all_numbers(stats):
    """45개 모든 번호에 점수 부여"""
    scores = {}
    for num in range(1, 46):
        scores[num] = score_number(num, stats)
    return scores


def generate_combinations_v2(data, max_combinations=500):
    """
    새로운 방식 조합 생성

    1. firstend로 (첫수, 끝수) 쌍 선택
    2. 각 번호에 인사이트 점수 부여
    3. 상위 점수 번호로 중간 4개(ball2-5) 선택
    4. 조합 생성 및 추가 점수 계산
    """
    stats = get_recent_stats(data, 50)

    # 1단계: (첫수, 끝수) 쌍 선택
    print("\n[1단계] firstend 쌍 선택")
    firstend_pairs = select_firstend_pairs(stats, top_ratio=0.6)
    print(f"  선택된 쌍: {len(firstend_pairs)}개")
    print(f"  상위 5개: {[(p['first'], p['last'], p['score']) for p in firstend_pairs[:5]]}")

    # 2단계: 모든 번호 점수 계산
    print("\n[2단계] 45개 번호 점수화")
    number_scores = score_all_numbers(stats)

    # 점수순 정렬
    sorted_numbers = sorted(range(1, 46), key=lambda x: -number_scores[x])
    print(f"  상위 10개: {[(n, number_scores[n]) for n in sorted_numbers[:10]]}")

    # 3단계: 조합 생성
    print("\n[3단계] 조합 생성")
    all_combos = []

    for pair in firstend_pairs[:100]:  # 상위 100개 쌍만 사용
        first = pair['first']
        last = pair['last']
        pair_score = pair['score']

        # 중간 4개 번호 후보 (first < x < last)
        mid_candidates = [n for n in sorted_numbers if first < n < last][:20]

        # 중간 4개 조합 생성
        for mid_combo in combinations(mid_candidates, 4):
            balls = sorted([first] + list(mid_combo) + [last])

            # 조합 점수 계산
            combo_score = pair_score
            for b in mid_combo:
                combo_score += number_scores[b]

            # 추가 인사이트 점수
            combo_score += score_combination_bonus(balls, stats)

            all_combos.append({
                'balls': balls,
                'score': combo_score
            })

    print(f"  생성된 조합: {len(all_combos):,}개")

    # 중복 제거
    seen = set()
    unique = []
    for c in all_combos:
        key = tuple(c['balls'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    print(f"  중복 제거 후: {len(unique):,}개")

    # 점수순 정렬
    unique.sort(key=lambda x: -x['score'])

    return unique[:max_combinations]


def score_combination_bonus(balls, stats):
    """조합 전체에 대한 추가 점수"""
    score = 0
    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # 8_sum: 합계 121-160이 48.5%
    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
    elif 100 <= total_sum <= 180:
        score += 5

    # 6_consecutive: 연속수 1쌍이 40.9%
    consecutive = 0
    for i in range(5):
        if balls[i+1] - balls[i] == 1:
            consecutive += 1

    if consecutive == 1:
        score += 8
    elif consecutive == 0:
        score += 5
    elif consecutive == 2:
        score += 3

    # 9_range: 4개 구간 사용이 51.2%
    ranges = set()
    for b in balls:
        if b <= 9: ranges.add(0)
        elif b <= 19: ranges.add(1)
        elif b <= 29: ranges.add(2)
        elif b <= 39: ranges.add(3)
        else: ranges.add(4)

    if len(ranges) == 4:
        score += 8
    elif len(ranges) == 5:
        score += 5
    elif len(ranges) == 3:
        score += 3

    # 10_prime: 소수 1-2개가 66.7%
    prime_count = sum(1 for b in balls if b in PRIMES)
    if prime_count in [1, 2]:
        score += 10
    elif prime_count == 3:
        score += 5

    # 2_second: ball2가 10-19면 보너스
    if 10 <= ball2 <= 19:
        score += 5

    # 3_third: ball3가 10-19면 보너스
    if 10 <= ball3 <= 19:
        score += 5

    # 4_fourth: ball4가 20-29면 보너스
    if 20 <= ball4 <= 29:
        score += 5

    # 5_fifth: ball5가 30-39면 보너스
    if 30 <= ball5 <= 39:
        score += 8

    # 7_lastnum: ball6가 40-45면 보너스
    if 40 <= ball6 <= 45:
        score += 8

    return score


def save_combinations(combinations, target_round, output_path):
    """조합 저장"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
            'score', 'sum', 'prime_count', 'consecutive'
        ])

        for i, combo in enumerate(combinations, 1):
            balls = combo['balls']
            consecutive = sum(1 for j in range(5) if balls[j+1] - balls[j] == 1)
            prime_count = sum(1 for b in balls if b in PRIMES)

            writer.writerow([
                i,
                balls[0], balls[1], balls[2], balls[3], balls[4], balls[5],
                combo['score'],
                sum(balls),
                prime_count,
                consecutive
            ])

    print(f"\n저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='조합 생성기 v2')
    parser.add_argument('--round', type=int, required=True, help='예측 회차')
    parser.add_argument('--count', type=int, default=500, help='생성할 조합 수')
    args = parser.parse_args()

    print("=" * 70)
    print(f"조합 생성기 v2 - {args.round}회차 예측")
    print("=" * 70)
    print("\n전략:")
    print("  1. firstend 인사이트로 (첫수, 끝수) 쌍 선택")
    print("  2. 12개 인사이트로 45개 번호 점수화")
    print("  3. 점수순 상위 번호로 조합 생성")

    # 데이터 로드
    data = load_data()
    print(f"\n총 데이터: {len(data)}회차 ({data[0]['round']} ~ {data[-1]['round']})")

    # 조합 생성
    combinations = generate_combinations_v2(data, args.count)

    # 결과 출력
    print(f"\n[상위 10개 조합]")
    print("-" * 70)
    for i, combo in enumerate(combinations[:10], 1):
        balls = combo['balls']
        balls_str = '-'.join(f"{b:2d}" for b in balls)
        print(f"  {i:2d}. {balls_str} | 점수: {combo['score']} | 합계: {sum(balls)}")

    # 점수 분포
    scores = [c['score'] for c in combinations]
    print(f"\n[점수 분포]")
    print(f"  최고: {max(scores)}, 최저: {min(scores)}, 평균: {sum(scores)/len(scores):.1f}")

    # 저장
    output_path = OUTPUT_DIR / f"combinations_v2_{args.round}.csv"
    save_combinations(combinations, args.round, output_path)

    print("\n완료!")


if __name__ == "__main__":
    main()
