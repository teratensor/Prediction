"""
혼합 전략 백테스트

Top24 + Rest21 혼합 전략의 적중률 검증
"""

import csv
from pathlib import Path
from collections import Counter
import statistics

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data():
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
    recent = data[-n:]
    all_freq = Counter()
    for r in recent:
        for ball in r['balls']:
            all_freq[ball] += 1

    sorted_nums = sorted(range(1, 46), key=lambda x: (-all_freq.get(x, 0), x))
    top24 = set(sorted_nums[:24])
    rest21 = set(sorted_nums[24:])

    return top24, rest21, all_freq


def get_range(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def score_combination(balls, top24):
    score = 0
    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    if 1 <= ball1 <= 10: score += 10
    if 38 <= ball6 <= 45: score += 10
    if 10 <= ball2 <= 19: score += 8
    elif 1 <= ball2 <= 9: score += 6
    if 10 <= ball3 <= 19: score += 8
    elif 20 <= ball3 <= 29: score += 7
    if 20 <= ball4 <= 29: score += 8
    elif 30 <= ball4 <= 39: score += 7
    if 30 <= ball5 <= 39: score += 10
    elif 20 <= ball5 <= 29: score += 5

    consecutive_pairs = sum(1 for i in range(5) if balls[i+1] - balls[i] == 1)
    if consecutive_pairs == 1: score += 8
    elif consecutive_pairs == 0: score += 5
    elif consecutive_pairs == 2: score += 3

    if 40 <= ball6 <= 45: score += 10
    elif 30 <= ball6 <= 39: score += 5

    total_sum = sum(balls)
    if 121 <= total_sum <= 160: score += 10
    elif 100 <= total_sum <= 170: score += 5

    ranges_used = len(set(get_range(b) for b in balls))
    if ranges_used == 4: score += 8
    elif ranges_used == 5: score += 6
    elif ranges_used == 3: score += 4

    prime_count = sum(1 for b in balls if b in PRIMES)
    if prime_count in [1, 2]: score += 10
    elif prime_count == 3: score += 5

    top24_count = sum(1 for b in balls if b in top24)
    if 3 <= top24_count <= 4: score += 10
    elif top24_count == 5: score += 5

    return score, top24_count


def generate_mixed_combos(data, round_idx, max_combos=500):
    """혼합 전략으로 조합 생성"""
    past_data = data[:round_idx]
    if len(past_data) < 50:
        return []

    top24, rest21, all_freq = get_recent_stats(past_data, 50)

    pos_ranges = {
        0: range(1, 16),
        1: range(2, 26),
        2: range(5, 36),
        3: range(12, 43),
        4: range(20, 45),
        5: range(30, 46),
    }

    pos_top = {}
    pos_rest = {}
    for pos, num_range in pos_ranges.items():
        pos_top[pos] = sorted([n for n in num_range if n in top24],
                              key=lambda x: -all_freq.get(x, 0))
        pos_rest[pos] = sorted([n for n in num_range if n in rest21],
                               key=lambda x: -all_freq.get(x, 0))

    # 패턴: Top4+Rest2, Top3+Rest3
    patterns = []
    # Top4 패턴 (15가지)
    from itertools import combinations as comb
    for top_pos in comb(range(6), 4):
        rest_pos = [p for p in range(6) if p not in top_pos]
        patterns.append((4, list(top_pos), rest_pos))
    # Top3 패턴 (20가지)
    for top_pos in comb(range(6), 3):
        rest_pos = [p for p in range(6) if p not in top_pos]
        patterns.append((3, list(top_pos), rest_pos))

    all_combos = []
    top_count, rest_count = 8, 6

    for _, top_pos, rest_pos in patterns:
        candidates = {}
        for pos in range(6):
            if pos in top_pos:
                candidates[pos] = pos_top[pos][:top_count]
            else:
                candidates[pos] = pos_rest[pos][:rest_count]
            if not candidates[pos]:
                all_in_range = list(pos_ranges[pos])
                candidates[pos] = sorted(all_in_range, key=lambda x: -all_freq.get(x, 0))[:8]

        for b1 in candidates[0]:
            for b2 in candidates[1]:
                if b2 <= b1: continue
                for b3 in candidates[2]:
                    if b3 <= b2: continue
                    for b4 in candidates[3]:
                        if b4 <= b3: continue
                        for b5 in candidates[4]:
                            if b5 <= b4: continue
                            for b6 in candidates[5]:
                                if b6 <= b5: continue
                                balls = [b1, b2, b3, b4, b5, b6]
                                score, top24_cnt = score_combination(balls, top24)
                                all_combos.append({'balls': balls, 'score': score, 'top24_count': top24_cnt})

    # 중복 제거
    seen = set()
    unique = []
    for c in all_combos:
        key = tuple(c['balls'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique.sort(key=lambda x: -x['score'])
    return unique[:max_combos], top24


def run_backtest(data, start_round=900, combo_count=500, max_rounds=None):
    """백테스트 실행"""
    results = []

    start_idx = None
    for i, d in enumerate(data):
        if d['round'] >= start_round:
            start_idx = i
            break

    if start_idx is None:
        return []

    end_idx = len(data)
    if max_rounds:
        end_idx = min(start_idx + max_rounds, len(data))

    total = end_idx - start_idx
    print(f"백테스트: {data[start_idx]['round']}~{data[end_idx-1]['round']}회차 ({total}회)")
    print("=" * 70)

    for idx in range(start_idx, end_idx):
        current = data[idx]
        target_round = current['round']
        actual_balls = current['balls']

        combos, top24 = generate_mixed_combos(data, idx, combo_count)
        if not combos:
            continue

        # 실제 당첨번호의 Top24 개수
        actual_top24 = sum(1 for b in actual_balls if b in top24)

        match_counts = [len(set(c['balls']) & set(actual_balls)) for c in combos]
        best_match = max(match_counts)
        match_dist = Counter(match_counts)

        match_3plus = sum(1 for m in match_counts if m >= 3)
        match_4plus = sum(1 for m in match_counts if m >= 4)
        match_5plus = sum(1 for m in match_counts if m >= 5)

        results.append({
            'round': target_round,
            'actual_balls': actual_balls,
            'actual_top24': actual_top24,
            'best_match': best_match,
            'top1_match': match_counts[0],
            'match_3plus': match_3plus,
            'match_4plus': match_4plus,
            'match_5plus': match_5plus,
            'match_6': match_dist.get(6, 0),
            'total_combos': len(combos)
        })

        progress = (idx - start_idx + 1) / total * 100
        if (idx - start_idx + 1) % 20 == 0 or idx == end_idx - 1:
            print(f"[{progress:5.1f}%] {target_round}회: 당첨 Top24={actual_top24}개, "
                  f"최고={best_match}개, 3+={match_3plus}, 4+={match_4plus}, 5+={match_5plus}")

    return results


def analyze_results(results):
    """결과 분석"""
    print("\n" + "=" * 70)
    print("백테스트 결과")
    print("=" * 70)

    total = len(results)

    # 최고 적중 분포
    best_dist = Counter(r['best_match'] for r in results)
    print("\n[최고 적중 분포]")
    for m in sorted(best_dist.keys(), reverse=True):
        cnt = best_dist[m]
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {m}개: {cnt:3d}회 ({pct:5.1f}%) {bar}")

    # N개 이상 적중 비율
    r3 = sum(1 for r in results if r['best_match'] >= 3)
    r4 = sum(1 for r in results if r['best_match'] >= 4)
    r5 = sum(1 for r in results if r['best_match'] >= 5)
    r6 = sum(1 for r in results if r['best_match'] == 6)

    print(f"\n[N개 이상 적중 비율]")
    print(f"  3개+: {r3}/{total} ({r3/total*100:.1f}%)")
    print(f"  4개+: {r4}/{total} ({r4/total*100:.1f}%)")
    print(f"  5개+: {r5}/{total} ({r5/total*100:.1f}%)")
    print(f"  6개:  {r6}/{total} ({r6/total*100:.1f}%)")

    # 평균
    avg_best = statistics.mean(r['best_match'] for r in results)
    avg_3plus = statistics.mean(r['match_3plus'] for r in results)
    avg_4plus = statistics.mean(r['match_4plus'] for r in results)

    print(f"\n[평균]")
    print(f"  최고 적중: {avg_best:.2f}개")
    print(f"  3+ 조합 수: {avg_3plus:.1f}개/500개")
    print(f"  4+ 조합 수: {avg_4plus:.1f}개/500개")

    # 당첨번호 Top24 개수별 적중률
    print(f"\n[당첨번호 Top24 개수별 최고 적중]")
    by_top24 = {}
    for r in results:
        t = r['actual_top24']
        if t not in by_top24:
            by_top24[t] = []
        by_top24[t].append(r['best_match'])

    for t in sorted(by_top24.keys(), reverse=True):
        matches = by_top24[t]
        avg = sum(matches) / len(matches)
        max_m = max(matches)
        print(f"  Top24={t}개: 평균 {avg:.2f}개, 최고 {max_m}개 ({len(matches)}회)")

    return {
        'total': total,
        'pct_3plus': r3 / total * 100,
        'pct_4plus': r4 / total * 100,
        'pct_5plus': r5 / total * 100,
        'avg_best': avg_best
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=900)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--max-rounds', type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("혼합 전략 백테스트")
    print("=" * 70)

    data = load_data()
    print(f"데이터: {len(data)}회차")

    results = run_backtest(data, args.start, args.count, args.max_rounds)

    if results:
        analyze_results(results)


if __name__ == "__main__":
    main()
