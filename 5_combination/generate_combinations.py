"""
인사이트 기반 조합 생성기 - 순차 빌드 방식

ord1부터 ord6까지 순서대로 인사이트를 적용하여 조합 생성

인사이트 적용 순서:
1. 1_firstend: 첫수/끝수 범위 결정
2. 1_second: ball2 범위 및 소수 패턴
3. 2_third: ball3 범위 및 소수 패턴
4. 3_fourth: ball4 범위 및 소수 패턴
5. 4_fifth: ball5 범위 (sum 기반)
6. 5_consecutive: 연속수 패턴 적용
7. 6_lastnum: ball6 범위 확정
8. 7_sum: 합계 범위 검증
9. 8_range: 구간 코드 검증
10. 9_prime: 소수 개수 검증
11. 10_shortcode: shortcode 패턴 검증
12. 11_onehot: 핫/콜드 비트 가중치
"""

import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import random

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
BACKTEST_PATH = Path(__file__).parent.parent / "3_backtest" / "backtest_results.csv"
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
                'ords': [int(row[f'ord{i}']) for i in range(1, 7)],
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
                'o_table': {i: int(row[f'o{i}']) for i in range(1, 46)}
            })
    return results


def get_recent_stats(data, n=50):
    """최근 N회차 통계"""
    recent = data[-n:]

    # 각 포지션별 빈도
    pos_freq = {pos: Counter() for pos in range(6)}
    for r in recent:
        for pos, ball in enumerate(r['balls']):
            pos_freq[pos][ball] += 1

    # 전체 번호 빈도
    all_freq = Counter()
    for r in recent:
        for ball in r['balls']:
            all_freq[ball] += 1

    return pos_freq, all_freq


def get_range(n):
    """구간 번호 반환 (0-4)"""
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def is_prime(n):
    return n in PRIMES


def calculate_sum(balls):
    return sum(balls)


def get_shortcode(ords, balls, o_table):
    """shortcode 계산 (Top24/Mid14/Rest7)"""
    # o_table에서 Top24, Mid14, Rest7 구분
    top24 = set(list(o_table.keys())[:24])  # ord 1-24
    mid14 = set(list(o_table.keys())[24:38])  # ord 25-38
    rest7 = set(list(o_table.keys())[38:])  # ord 39-45

    ord_code = [0, 0, 0]
    ball_code = [0, 0, 0]

    for ord_val in ords:
        if ord_val <= 24:
            ord_code[0] += 1
        elif ord_val <= 38:
            ord_code[1] += 1
        else:
            ord_code[2] += 1

    # ball의 세그먼트는 o_table 기준
    for ball in balls:
        ord_of_ball = None
        for k, v in o_table.items():
            if v == ball:
                ord_of_ball = k
                break
        if ord_of_ball:
            if ord_of_ball <= 24:
                ball_code[0] += 1
            elif ord_of_ball <= 38:
                ball_code[1] += 1
            else:
                ball_code[2] += 1

    return ''.join(map(str, ord_code)) + ''.join(map(str, ball_code))


def score_combination(balls, insights, o_table):
    """조합에 대한 점수 계산 및 적용된 인사이트 기록"""
    score = 0
    applied = []

    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # 1. firstend: 첫수/끝수 범위
    if 1 <= ball1 <= 10:
        score += 10
        applied.append("firstend:ball1_ok")
    if 38 <= ball6 <= 45:
        score += 10
        applied.append("firstend:ball6_ok")

    # 2. second: ball2 범위 (10-19가 48.3%)
    if 10 <= ball2 <= 19:
        score += 8
        applied.append("second:range1")
    elif 1 <= ball2 <= 9:
        score += 6
        applied.append("second:range0")

    # ball1-ball2 소수 패턴 (NN이 36.9%)
    if not is_prime(ball1) and not is_prime(ball2):
        score += 5
        applied.append("second:NN_pattern")

    # 3. third: ball3 범위 (10-19가 45.4%, 20-29가 36.7%)
    if 10 <= ball3 <= 19:
        score += 8
        applied.append("third:range1")
    elif 20 <= ball3 <= 29:
        score += 7
        applied.append("third:range2")

    # 4. fourth: ball4 범위 (20-29가 42.7%, 30-39가 34.8%)
    if 20 <= ball4 <= 29:
        score += 8
        applied.append("fourth:range2")
    elif 30 <= ball4 <= 39:
        score += 7
        applied.append("fourth:range3")

    # ball1,2,3 소수 개수와 ball4 관계
    prime_count_123 = sum(1 for b in [ball1, ball2, ball3] if is_prime(b))
    if prime_count_123 <= 1:
        score += 3
        applied.append("fourth:low_prime_ok")

    # 5. fifth: ball5 범위 (30-39가 54.6%)
    if 30 <= ball5 <= 39:
        score += 10
        applied.append("fifth:range3")
    elif 20 <= ball5 <= 29:
        score += 5
        applied.append("fifth:range2")

    # ball1~4 합계 기반 ball5 예측
    sum14 = ball1 + ball2 + ball3 + ball4
    if sum14 <= 40 and 18 <= ball5 <= 33:
        score += 5
        applied.append("fifth:sum14_low")
    elif 41 <= sum14 <= 70 and 22 <= ball5 <= 38:
        score += 5
        applied.append("fifth:sum14_mid")
    elif sum14 >= 71 and 30 <= ball5 <= 41:
        score += 5
        applied.append("fifth:sum14_high")

    # 6. consecutive: 연속수 패턴 (52.2%가 연속수 있음)
    consecutive_pairs = 0
    for i in range(5):
        if balls[i+1] - balls[i] == 1:
            consecutive_pairs += 1

    if consecutive_pairs == 1:
        score += 8
        applied.append("consecutive:1pair")
    elif consecutive_pairs == 0:
        score += 5
        applied.append("consecutive:0pair")
    elif consecutive_pairs == 2:
        score += 3
        applied.append("consecutive:2pair")

    # 7. lastnum: ball6 범위 (40-45가 57.8%)
    if 40 <= ball6 <= 45:
        score += 10
        applied.append("lastnum:range4")
    elif 30 <= ball6 <= 39:
        score += 5
        applied.append("lastnum:range3")

    # 8. sum: 합계 범위 (121-160이 48.5%)
    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
        applied.append("sum:121-160")
    elif 100 <= total_sum <= 170:
        score += 5
        applied.append("sum:100-170")

    # 9. range: 구간 코드 (4개 구간 사용이 51.2%)
    ranges_used = len(set(get_range(b) for b in balls))
    if ranges_used == 4:
        score += 8
        applied.append("range:4segments")
    elif ranges_used == 5:
        score += 6
        applied.append("range:5segments")
    elif ranges_used == 3:
        score += 4
        applied.append("range:3segments")

    # 10. prime: 소수 개수 (1-2개가 66.7%)
    prime_count = sum(1 for b in balls if is_prime(b))
    if prime_count == 1 or prime_count == 2:
        score += 10
        applied.append(f"prime:{prime_count}primes")
    elif prime_count == 3:
        score += 5
        applied.append("prime:3primes")

    # 11. shortcode: 패턴 검증 (Top24에서 3-4개)
    # o_table 기반 검증
    ord_top24_count = 0
    for ball in balls:
        for ord_val, ball_val in o_table.items():
            if ball_val == ball and ord_val <= 24:
                ord_top24_count += 1
                break

    if 3 <= ord_top24_count <= 4:
        score += 10
        applied.append(f"shortcode:top24_{ord_top24_count}")
    elif ord_top24_count == 5:
        score += 5
        applied.append("shortcode:top24_5")

    # 12. onehot: 핫 비트 가중치 (29, 17, 27이 핫)
    hot_bits = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
    cold_bits = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    hot_count = sum(1 for b in balls if b in hot_bits)
    cold_count = sum(1 for b in balls if b in cold_bits)

    if hot_count >= 2:
        score += hot_count * 2
        applied.append(f"onehot:hot_{hot_count}")
    if cold_count >= 2:
        score -= cold_count
        applied.append(f"onehot:cold_{cold_count}")

    return score, applied


def generate_candidates(data, target_round):
    """후보 번호 생성"""
    # 최근 데이터에서 o_table 가져오기
    latest = data[-1]
    o_table = latest['o_table']

    # 최근 50회 통계
    pos_freq, all_freq = get_recent_stats(data, 50)

    # 각 포지션별 후보 번호 선정 (인사이트 기반)
    candidates = {
        'ball1': [],  # 1-10 범위, 빈도 높은 순
        'ball2': [],  # 1-20 범위
        'ball3': [],  # 10-30 범위
        'ball4': [],  # 20-40 범위
        'ball5': [],  # 25-42 범위
        'ball6': [],  # 30-45 범위
    }

    # ball1: 1-10 범위에서 빈도순
    for num in range(1, 11):
        candidates['ball1'].append((num, pos_freq[0].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball1'].sort(key=lambda x: -x[1])

    # ball2: 1-25 범위
    for num in range(2, 26):
        candidates['ball2'].append((num, pos_freq[1].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball2'].sort(key=lambda x: -x[1])

    # ball3: 5-35 범위
    for num in range(5, 36):
        candidates['ball3'].append((num, pos_freq[2].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball3'].sort(key=lambda x: -x[1])

    # ball4: 15-42 범위
    for num in range(15, 43):
        candidates['ball4'].append((num, pos_freq[3].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball4'].sort(key=lambda x: -x[1])

    # ball5: 25-44 범위
    for num in range(25, 45):
        candidates['ball5'].append((num, pos_freq[4].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball5'].sort(key=lambda x: -x[1])

    # ball6: 30-45 범위
    for num in range(30, 46):
        candidates['ball6'].append((num, pos_freq[5].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball6'].sort(key=lambda x: -x[1])

    return candidates, o_table


def build_combinations(candidates, o_table, max_combinations=500):
    """순차 빌드 방식으로 조합 생성"""
    all_combos = []

    # 상위 후보만 사용
    b1_candidates = [x[0] for x in candidates['ball1'][:8]]
    b2_candidates = [x[0] for x in candidates['ball2'][:12]]
    b3_candidates = [x[0] for x in candidates['ball3'][:15]]
    b4_candidates = [x[0] for x in candidates['ball4'][:15]]
    b5_candidates = [x[0] for x in candidates['ball5'][:12]]
    b6_candidates = [x[0] for x in candidates['ball6'][:10]]

    print(f"후보 번호:")
    print(f"  ball1: {b1_candidates}")
    print(f"  ball2: {b2_candidates}")
    print(f"  ball3: {b3_candidates}")
    print(f"  ball4: {b4_candidates}")
    print(f"  ball5: {b5_candidates}")
    print(f"  ball6: {b6_candidates}")

    # 조합 생성 및 점수 계산
    combo_count = 0
    for b1 in b1_candidates:
        for b2 in b2_candidates:
            if b2 <= b1:
                continue
            for b3 in b3_candidates:
                if b3 <= b2:
                    continue
                for b4 in b4_candidates:
                    if b4 <= b3:
                        continue
                    for b5 in b5_candidates:
                        if b5 <= b4:
                            continue
                        for b6 in b6_candidates:
                            if b6 <= b5:
                                continue

                            balls = [b1, b2, b3, b4, b5, b6]
                            score, applied = score_combination(balls, None, o_table)
                            all_combos.append({
                                'balls': balls,
                                'score': score,
                                'applied': applied
                            })
                            combo_count += 1

    print(f"\n생성된 총 조합 수: {combo_count:,}")

    # 점수순 정렬
    all_combos.sort(key=lambda x: -x['score'])

    return all_combos[:max_combinations]


def save_results(combinations, target_round, output_path):
    """결과 저장"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
            'score', 'sum', 'prime_count', 'consecutive_pairs', 'range_code',
            'insights_applied'
        ])

        for rank, combo in enumerate(combinations, 1):
            balls = combo['balls']

            # 추가 정보 계산
            total_sum = sum(balls)
            prime_count = sum(1 for b in balls if is_prime(b))

            consec_pairs = 0
            for i in range(5):
                if balls[i+1] - balls[i] == 1:
                    consec_pairs += 1

            range_code = ''.join(str(get_range(b)) for b in balls)

            writer.writerow([
                rank,
                balls[0], balls[1], balls[2], balls[3], balls[4], balls[5],
                combo['score'],
                total_sum,
                prime_count,
                consec_pairs,
                range_code,
                '|'.join(combo['applied'])
            ])

    print(f"\n결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='인사이트 기반 조합 생성기')
    parser.add_argument('--round', type=int, help='예측 대상 회차')
    parser.add_argument('--count', type=int, default=500, help='생성할 조합 수')
    args = parser.parse_args()

    # 데이터 로드
    data = load_data()
    latest_round = data[-1]['round']

    target_round = args.round if args.round else latest_round + 1

    print("=" * 70)
    print("인사이트 기반 조합 생성기 (순차 빌드 방식)")
    print("=" * 70)
    print(f"\n데이터: {len(data)}회차 ({data[0]['round']} ~ {latest_round})")
    print(f"예측 대상: {target_round}회차")
    print(f"생성 조합 수: {args.count}개")

    # 후보 생성
    print("\n" + "=" * 70)
    print("1단계: 후보 번호 선정")
    print("=" * 70)
    candidates, o_table = generate_candidates(data, target_round)

    # 조합 생성
    print("\n" + "=" * 70)
    print("2단계: 조합 생성 및 점수 계산")
    print("=" * 70)
    combinations = build_combinations(candidates, o_table, args.count)

    # 결과 저장
    print("\n" + "=" * 70)
    print("3단계: 결과 저장")
    print("=" * 70)

    output_path = OUTPUT_DIR / f"combinations_round{target_round}.csv"
    save_results(combinations, target_round, output_path)

    # 상위 10개 출력
    print("\n" + "=" * 70)
    print("상위 10개 조합")
    print("=" * 70)
    print(f"{'순위':>4} {'번호':^25} {'점수':>5} {'합계':>5} {'소수':>4} {'연속':>4} {'구간코드':>8}")
    print("-" * 70)

    for i, combo in enumerate(combinations[:10], 1):
        balls = combo['balls']
        total_sum = sum(balls)
        prime_count = sum(1 for b in balls if is_prime(b))
        consec_pairs = sum(1 for j in range(5) if balls[j+1] - balls[j] == 1)
        range_code = ''.join(str(get_range(b)) for b in balls)

        balls_str = ', '.join(f"{b:2d}" for b in balls)
        print(f"{i:4d} [{balls_str}] {combo['score']:5d} {total_sum:5d} {prime_count:4d} {consec_pairs:4d} {range_code:>8}")

    print("\n" + "=" * 70)
    print("적용된 인사이트 (상위 1위 조합):")
    print("=" * 70)
    for insight in combinations[0]['applied']:
        print(f"  - {insight}")

    print("\n완료!")


if __name__ == "__main__":
    main()
