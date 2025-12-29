"""
혼합 전략 조합 생성기

Top24에서 3-4개 + Rest에서 2-3개를 혼합하여 조합 생성
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
    """최근 N회차 통계 - 전체 빈도 기준 순위"""
    recent = data[-n:]

    all_freq = Counter()
    for r in recent:
        for ball in r['balls']:
            all_freq[ball] += 1

    # 빈도순 정렬 (동점시 번호 오름차순)
    sorted_nums = sorted(range(1, 46), key=lambda x: (-all_freq.get(x, 0), x))

    # Top24, Rest21 분리
    top24 = set(sorted_nums[:24])
    rest21 = set(sorted_nums[24:])

    return top24, rest21, all_freq


def get_range(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def score_combination(balls, o_table, top24):
    """조합에 대한 점수 계산"""
    score = 0

    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # 1. firstend (20점)
    if 1 <= ball1 <= 10:
        score += 10
    if 38 <= ball6 <= 45:
        score += 10

    # 2. second (8점)
    if 10 <= ball2 <= 19:
        score += 8
    elif 1 <= ball2 <= 9:
        score += 6

    # 3. third (8점)
    if 10 <= ball3 <= 19:
        score += 8
    elif 20 <= ball3 <= 29:
        score += 7

    # 4. fourth (8점)
    if 20 <= ball4 <= 29:
        score += 8
    elif 30 <= ball4 <= 39:
        score += 7

    # 5. fifth (10점)
    if 30 <= ball5 <= 39:
        score += 10
    elif 20 <= ball5 <= 29:
        score += 5

    # 6. consecutive (8점)
    consecutive_pairs = 0
    for i in range(5):
        if balls[i+1] - balls[i] == 1:
            consecutive_pairs += 1

    if consecutive_pairs == 1:
        score += 8
    elif consecutive_pairs == 0:
        score += 5
    elif consecutive_pairs == 2:
        score += 3

    # 7. lastnum (10점)
    if 40 <= ball6 <= 45:
        score += 10
    elif 30 <= ball6 <= 39:
        score += 5

    # 8. sum (10점)
    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
    elif 100 <= total_sum <= 170:
        score += 5

    # 9. range (8점)
    ranges_used = len(set(get_range(b) for b in balls))
    if ranges_used == 4:
        score += 8
    elif ranges_used == 5:
        score += 6
    elif ranges_used == 3:
        score += 4

    # 10. prime (10점)
    prime_count = sum(1 for b in balls if b in PRIMES)
    if prime_count == 1 or prime_count == 2:
        score += 10
    elif prime_count == 3:
        score += 5

    # 11. shortcode - Top24 개수 (10점)
    top24_count = sum(1 for b in balls if b in top24)
    if 3 <= top24_count <= 4:
        score += 10
    elif top24_count == 5:
        score += 5

    return score, top24_count


def generate_mixed_combinations(data, max_combinations=500):
    """혼합 전략으로 조합 생성"""

    top24, rest21, all_freq = get_recent_stats(data, 50)
    latest = data[-1]
    o_table = latest['o_table']

    print(f"\n[세그먼트 분류]")
    print(f"  Top24 (빈도 상위): {sorted(top24)}")
    print(f"  Rest21 (빈도 하위): {sorted(rest21)}")

    # 포지션별 범위 정의
    pos_ranges = {
        0: range(1, 16),   # ball1: 1-15
        1: range(2, 26),   # ball2: 2-25
        2: range(5, 36),   # ball3: 5-35
        3: range(12, 43),  # ball4: 12-42
        4: range(20, 45),  # ball5: 20-44
        5: range(30, 46),  # ball6: 30-45
    }

    # 각 포지션에서 Top24/Rest21 후보 분리
    pos_top = {}
    pos_rest = {}

    for pos, num_range in pos_ranges.items():
        pos_top[pos] = sorted([n for n in num_range if n in top24],
                              key=lambda x: -all_freq.get(x, 0))
        pos_rest[pos] = sorted([n for n in num_range if n in rest21],
                               key=lambda x: -all_freq.get(x, 0))

    print(f"\n[포지션별 후보]")
    for pos in range(6):
        print(f"  ball{pos+1}: Top={pos_top[pos][:8]}... Rest={pos_rest[pos][:6]}...")

    # 조합 생성: Top24에서 3-4개, Rest에서 2-3개
    all_combos = []

    # 패턴 정의: (Top24 선택 포지션 수, Rest 선택 포지션 수)
    # 3-4개를 Top24에서, 2-3개를 Rest에서

    patterns = [
        # Top24에서 4개 선택하는 패턴 (C(6,4) = 15가지)
        (4, [0,1,2,3], [4,5]),
        (4, [0,1,2,4], [3,5]),
        (4, [0,1,2,5], [3,4]),
        (4, [0,1,3,4], [2,5]),
        (4, [0,1,3,5], [2,4]),
        (4, [0,1,4,5], [2,3]),
        (4, [0,2,3,4], [1,5]),
        (4, [0,2,3,5], [1,4]),
        (4, [0,2,4,5], [1,3]),
        (4, [0,3,4,5], [1,2]),
        (4, [1,2,3,4], [0,5]),
        (4, [1,2,3,5], [0,4]),
        (4, [1,2,4,5], [0,3]),
        (4, [1,3,4,5], [0,2]),
        (4, [2,3,4,5], [0,1]),

        # Top24에서 3개 선택하는 패턴 (C(6,3) = 20가지)
        (3, [0,1,2], [3,4,5]),
        (3, [0,1,3], [2,4,5]),
        (3, [0,1,4], [2,3,5]),
        (3, [0,1,5], [2,3,4]),
        (3, [0,2,3], [1,4,5]),
        (3, [0,2,4], [1,3,5]),
        (3, [0,2,5], [1,3,4]),
        (3, [0,3,4], [1,2,5]),
        (3, [0,3,5], [1,2,4]),
        (3, [0,4,5], [1,2,3]),
        (3, [1,2,3], [0,4,5]),
        (3, [1,2,4], [0,3,5]),
        (3, [1,2,5], [0,3,4]),
        (3, [1,3,4], [0,2,5]),
        (3, [1,3,5], [0,2,4]),
        (3, [1,4,5], [0,2,3]),
        (3, [2,3,4], [0,1,5]),
        (3, [2,3,5], [0,1,4]),
        (3, [2,4,5], [0,1,3]),
        (3, [3,4,5], [0,1,2]),
    ]

    print(f"\n[조합 생성 중...]")

    # 각 포지션에서 사용할 후보 수
    top_count = 8   # Top24에서 상위 8개
    rest_count = 6  # Rest21에서 상위 6개

    for pattern_idx, (top_n, top_pos, rest_pos) in enumerate(patterns):
        # 각 포지션별 후보 선택
        candidates = {}
        for pos in range(6):
            if pos in top_pos:
                candidates[pos] = pos_top[pos][:top_count]
            else:
                candidates[pos] = pos_rest[pos][:rest_count]

            # 후보가 없으면 전체에서 선택
            if not candidates[pos]:
                all_in_range = list(pos_ranges[pos])
                candidates[pos] = sorted(all_in_range, key=lambda x: -all_freq.get(x, 0))[:8]

        # 조합 생성
        for b1 in candidates[0]:
            for b2 in candidates[1]:
                if b2 <= b1:
                    continue
                for b3 in candidates[2]:
                    if b3 <= b2:
                        continue
                    for b4 in candidates[3]:
                        if b4 <= b3:
                            continue
                        for b5 in candidates[4]:
                            if b5 <= b4:
                                continue
                            for b6 in candidates[5]:
                                if b6 <= b5:
                                    continue

                                balls = [b1, b2, b3, b4, b5, b6]
                                score, top24_count = score_combination(balls, o_table, top24)

                                all_combos.append({
                                    'balls': balls,
                                    'score': score,
                                    'top24_count': top24_count,
                                    'pattern': f"Top{top_n}"
                                })

    print(f"  총 생성 조합: {len(all_combos):,}개")

    # 중복 제거
    seen = set()
    unique_combos = []
    for c in all_combos:
        key = tuple(c['balls'])
        if key not in seen:
            seen.add(key)
            unique_combos.append(c)

    print(f"  중복 제거 후: {len(unique_combos):,}개")

    # 점수순 정렬
    unique_combos.sort(key=lambda x: -x['score'])

    return unique_combos[:max_combinations], top24, rest21


def save_combinations(combinations, target_round, output_path):
    """조합 저장"""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'rank', 'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6',
            'score', 'sum', 'top24_count', 'pattern'
        ])

        for i, combo in enumerate(combinations, 1):
            balls = combo['balls']
            writer.writerow([
                i,
                balls[0], balls[1], balls[2], balls[3], balls[4], balls[5],
                combo['score'],
                sum(balls),
                combo['top24_count'],
                combo['pattern']
            ])

    print(f"\n저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='혼합 전략 조합 생성기')
    parser.add_argument('--round', type=int, required=True, help='예측 회차')
    parser.add_argument('--count', type=int, default=500, help='생성할 조합 수')
    args = parser.parse_args()

    print("=" * 70)
    print(f"혼합 전략 조합 생성기 - {args.round}회차 예측")
    print("=" * 70)

    # 데이터 로드
    data = load_data()
    print(f"\n총 데이터: {len(data)}회차 ({data[0]['round']} ~ {data[-1]['round']})")

    # 조합 생성
    combinations, top24, rest21 = generate_mixed_combinations(data, args.count)

    # Top24 개수 분포
    top24_dist = Counter(c['top24_count'] for c in combinations)
    print(f"\n[상위 {args.count}개 Top24 개수 분포]")
    for cnt in sorted(top24_dist.keys(), reverse=True):
        freq = top24_dist[cnt]
        pct = freq / len(combinations) * 100
        print(f"  {cnt}개: {freq}개 ({pct:.1f}%)")

    # 점수 분포
    scores = [c['score'] for c in combinations]
    print(f"\n[점수 분포]")
    print(f"  최고: {max(scores)}점")
    print(f"  최저: {min(scores)}점")
    print(f"  평균: {sum(scores)/len(scores):.1f}점")

    # 상위 10개 출력
    print(f"\n[상위 10개 조합]")
    print("-" * 70)
    for i, combo in enumerate(combinations[:10], 1):
        balls = combo['balls']
        balls_str = '-'.join(f"{b:2d}" for b in balls)
        print(f"  {i:2d}. {balls_str} | 점수: {combo['score']} | Top24: {combo['top24_count']}개 | {combo['pattern']}")

    # 저장
    output_path = OUTPUT_DIR / f"combinations_mixed_{args.round}.csv"
    save_combinations(combinations, args.round, output_path)

    print("\n완료!")


if __name__ == "__main__":
    main()
