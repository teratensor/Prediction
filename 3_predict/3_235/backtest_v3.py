"""
백테스트 v3 - 혼합 전략 + firstend 개선

전략:
1. 혼합 전략 (Top24 + Rest21) 기반
2. firstend에서 50% seen + 50% unseen 적용
3. 포지션별 후보에 firstend 확률 반영
"""

import csv
from pathlib import Path
from collections import Counter
from itertools import combinations
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

    # 빈도순 정렬
    sorted_nums = sorted(range(1, 46), key=lambda x: (-all_freq.get(x, 0), x))
    top24 = set(sorted_nums[:24])
    rest21 = set(sorted_nums[24:])

    # 첫수/끝수 빈도
    first_freq = Counter(r['balls'][0] for r in recent)
    last_freq = Counter(r['balls'][5] for r in recent)
    pair_freq = Counter((r['balls'][0], r['balls'][5]) for r in recent)

    return {
        'all_freq': all_freq,
        'top24': top24,
        'rest21': rest21,
        'first_freq': first_freq,
        'last_freq': last_freq,
        'pair_freq': pair_freq,
        'recent': recent
    }


def get_range(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def get_firstend_candidates(stats):
    """
    firstend 후보 생성 - 50% seen + 50% unseen
    """
    first_freq = stats['first_freq']
    last_freq = stats['last_freq']
    pair_freq = stats['pair_freq']
    recent = stats['recent']

    recent_3 = set()
    for r in recent[-3:]:
        recent_3.update(r['balls'])

    seen_pairs = set(pair_freq.keys())

    # 1. seen 첫수/끝수 후보 (학습 데이터에서 나온 것)
    seen_firsts = set()
    seen_lasts = set()
    for (f, l) in seen_pairs:
        seen_firsts.add(f)
        seen_lasts.add(l)

    # 2. unseen 후보 (나오지 않았지만 가능성 있는 것)
    # 첫수: 1-15 중 seen에 없는 것
    unseen_firsts = set(range(1, 16)) - seen_firsts
    # 끝수: 30-45 중 seen에 없는 것
    unseen_lasts = set(range(30, 46)) - seen_lasts

    # 3. 점수 기반 정렬
    def score_first(n):
        score = first_freq.get(n, 0) * 3
        if 1 <= n <= 10: score += 5
        if n in recent_3: score -= 3
        return score

    def score_last(n):
        score = last_freq.get(n, 0) * 3
        if 38 <= n <= 45: score += 8
        elif 35 <= n <= 37: score += 4
        if n in recent_3: score -= 3
        return score

    # 상위 50% seen + 상위 50% unseen
    seen_firsts_sorted = sorted(seen_firsts, key=lambda x: -score_first(x))
    seen_lasts_sorted = sorted(seen_lasts, key=lambda x: -score_last(x))

    unseen_firsts_sorted = sorted(unseen_firsts, key=lambda x: -score_first(x))
    unseen_lasts_sorted = sorted(unseen_lasts, key=lambda x: -score_last(x))

    # 각각 50%씩 선택
    n_seen_first = max(len(seen_firsts_sorted) // 2, 5)
    n_unseen_first = max(len(unseen_firsts_sorted) // 2, 3)
    n_seen_last = max(len(seen_lasts_sorted) // 2, 5)
    n_unseen_last = max(len(unseen_lasts_sorted) // 2, 3)

    first_candidates = list(seen_firsts_sorted[:n_seen_first]) + list(unseen_firsts_sorted[:n_unseen_first])
    last_candidates = list(seen_lasts_sorted[:n_seen_last]) + list(unseen_lasts_sorted[:n_unseen_last])

    return set(first_candidates), set(last_candidates)


def score_combination(balls, top24, first_candidates, last_candidates):
    score = 0
    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # firstend 보너스 (새로운 전략)
    if ball1 in first_candidates:
        score += 10
    if ball6 in last_candidates:
        score += 10

    # 기존 포지션 점수
    if 1 <= ball1 <= 10: score += 8
    if 10 <= ball2 <= 19: score += 8
    elif 1 <= ball2 <= 9: score += 6
    if 10 <= ball3 <= 19: score += 8
    elif 20 <= ball3 <= 29: score += 7
    if 20 <= ball4 <= 29: score += 8
    elif 30 <= ball4 <= 39: score += 7
    if 30 <= ball5 <= 39: score += 10
    elif 20 <= ball5 <= 29: score += 5
    if 40 <= ball6 <= 45: score += 10
    elif 30 <= ball6 <= 39: score += 5

    # 연속수
    consecutive_pairs = sum(1 for i in range(5) if balls[i+1] - balls[i] == 1)
    if consecutive_pairs == 1: score += 8
    elif consecutive_pairs == 0: score += 5
    elif consecutive_pairs == 2: score += 3

    # 합계
    total_sum = sum(balls)
    if 121 <= total_sum <= 160: score += 10
    elif 100 <= total_sum <= 170: score += 5

    # 구간 분포
    ranges_used = len(set(get_range(b) for b in balls))
    if ranges_used == 4: score += 8
    elif ranges_used == 5: score += 6
    elif ranges_used == 3: score += 4

    # 소수
    prime_count = sum(1 for b in balls if b in PRIMES)
    if prime_count in [1, 2]: score += 10
    elif prime_count == 3: score += 5

    # Top24 개수
    top24_count = sum(1 for b in balls if b in top24)
    if 3 <= top24_count <= 4: score += 10
    elif top24_count == 5: score += 5

    return score, top24_count


def generate_combos_for_round(data, round_idx, max_combos=500):
    past_data = data[:round_idx]
    if len(past_data) < 50:
        return [], set()

    stats = get_recent_stats(past_data, 50)
    top24 = stats['top24']
    rest21 = stats['rest21']
    all_freq = stats['all_freq']

    # firstend 후보 (50% seen + 50% unseen)
    first_candidates, last_candidates = get_firstend_candidates(stats)

    # 포지션별 범위
    pos_ranges = {
        0: range(1, 16),
        1: range(2, 26),
        2: range(5, 36),
        3: range(12, 43),
        4: range(20, 45),
        5: range(30, 46),
    }

    # 포지션별 Top24/Rest21 후보
    pos_top = {}
    pos_rest = {}
    for pos, num_range in pos_ranges.items():
        pos_top[pos] = sorted([n for n in num_range if n in top24],
                              key=lambda x: -all_freq.get(x, 0))
        pos_rest[pos] = sorted([n for n in num_range if n in rest21],
                               key=lambda x: -all_freq.get(x, 0))

    # ball1에 firstend 후보 우선 반영
    ball1_candidates = []
    for n in first_candidates:
        if n in pos_ranges[0]:
            ball1_candidates.append(n)
    ball1_candidates = sorted(ball1_candidates, key=lambda x: -all_freq.get(x, 0))[:10]
    # 나머지 추가
    for n in pos_top[0][:8]:
        if n not in ball1_candidates:
            ball1_candidates.append(n)

    # ball6에 firstend 후보 우선 반영
    ball6_candidates = []
    for n in last_candidates:
        if n in pos_ranges[5]:
            ball6_candidates.append(n)
    ball6_candidates = sorted(ball6_candidates, key=lambda x: -all_freq.get(x, 0))[:10]
    for n in pos_top[5][:8]:
        if n not in ball6_candidates:
            ball6_candidates.append(n)

    # 패턴 생성: Top4+Rest2, Top3+Rest3
    patterns = []
    for top_pos in combinations(range(6), 4):
        rest_pos = [p for p in range(6) if p not in top_pos]
        patterns.append((4, list(top_pos), rest_pos))
    for top_pos in combinations(range(6), 3):
        rest_pos = [p for p in range(6) if p not in top_pos]
        patterns.append((3, list(top_pos), rest_pos))

    all_combos = []
    top_count, rest_count = 8, 6

    for _, top_pos, rest_pos in patterns:
        candidates = {}
        for pos in range(6):
            if pos == 0:
                candidates[pos] = ball1_candidates[:top_count]
            elif pos == 5:
                candidates[pos] = ball6_candidates[:top_count]
            elif pos in top_pos:
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
                                score, top24_cnt = score_combination(balls, top24, first_candidates, last_candidates)
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

        combos, top24 = generate_combos_for_round(data, idx, combo_count)
        if not combos:
            continue

        actual_top24 = sum(1 for b in actual_balls if b in top24)

        match_counts = [len(set(c['balls']) & set(actual_balls)) for c in combos]
        best_match = max(match_counts)
        match_dist = Counter(match_counts)

        match_3plus = sum(1 for m in match_counts if m >= 3)
        match_4plus = sum(1 for m in match_counts if m >= 4)
        match_5plus = sum(1 for m in match_counts if m >= 5)

        # 첫수/끝수 적중
        first_hit = any(c['balls'][0] == actual_balls[0] for c in combos)
        last_hit = any(c['balls'][5] == actual_balls[5] for c in combos)

        results.append({
            'round': target_round,
            'actual_balls': actual_balls,
            'actual_top24': actual_top24,
            'best_match': best_match,
            'match_3plus': match_3plus,
            'match_4plus': match_4plus,
            'match_5plus': match_5plus,
            'first_hit': first_hit,
            'last_hit': last_hit,
            'total_combos': len(combos)
        })

        progress = (idx - start_idx + 1) / total * 100
        if (idx - start_idx + 1) % 20 == 0 or idx == end_idx - 1:
            print(f"[{progress:5.1f}%] {target_round}회: Top24={actual_top24}개, "
                  f"최고={best_match}개, 3+={match_3plus}, 4+={match_4plus}, "
                  f"첫수={first_hit}, 끝수={last_hit}")

    return results


def analyze_results(results):
    print("\n" + "=" * 70)
    print("백테스트 결과 (v3 - 혼합전략 + firstend 50/50)")
    print("=" * 70)

    total = len(results)

    best_dist = Counter(r['best_match'] for r in results)
    print("\n[최고 적중 분포]")
    for m in sorted(best_dist.keys(), reverse=True):
        cnt = best_dist[m]
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {m}개: {cnt:3d}회 ({pct:5.1f}%) {bar}")

    r3 = sum(1 for r in results if r['best_match'] >= 3)
    r4 = sum(1 for r in results if r['best_match'] >= 4)
    r5 = sum(1 for r in results if r['best_match'] >= 5)
    r6 = sum(1 for r in results if r['best_match'] == 6)

    print(f"\n[N개 이상 적중 비율]")
    print(f"  3개+: {r3}/{total} ({r3/total*100:.1f}%)")
    print(f"  4개+: {r4}/{total} ({r4/total*100:.1f}%)")
    print(f"  5개+: {r5}/{total} ({r5/total*100:.1f}%)")
    print(f"  6개:  {r6}/{total} ({r6/total*100:.1f}%)")

    first_hits = sum(1 for r in results if r['first_hit'])
    last_hits = sum(1 for r in results if r['last_hit'])

    print(f"\n[firstend 적중률]")
    print(f"  첫수 적중: {first_hits}/{total} ({first_hits/total*100:.1f}%)")
    print(f"  끝수 적중: {last_hits}/{total} ({last_hits/total*100:.1f}%)")

    avg_best = statistics.mean(r['best_match'] for r in results)
    avg_3plus = statistics.mean(r['match_3plus'] for r in results)
    avg_4plus = statistics.mean(r['match_4plus'] for r in results)

    print(f"\n[평균]")
    print(f"  최고 적중: {avg_best:.2f}개")
    print(f"  3+ 조합 수: {avg_3plus:.1f}개/500개")
    print(f"  4+ 조합 수: {avg_4plus:.1f}개/500개")

    return {
        'total': total,
        'pct_3plus': r3 / total * 100,
        'pct_4plus': r4 / total * 100,
        'pct_5plus': r5 / total * 100,
        'avg_best': avg_best,
        'first_hit_pct': first_hits / total * 100,
        'last_hit_pct': last_hits / total * 100
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=900)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--max-rounds', type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("백테스트 v3 - 혼합전략 + firstend 50/50")
    print("=" * 70)

    data = load_data()
    print(f"데이터: {len(data)}회차")

    results = run_backtest(data, args.start, args.count, args.max_rounds)

    if results:
        analyze_results(results)


if __name__ == "__main__":
    main()
