"""
백테스트 v2 - 새로운 방식

전략:
1. firstend 인사이트로 (첫수, 끝수) 쌍 선택
2. 12개 인사이트로 45개 번호 점수화
3. 점수순 상위 번호로 조합 생성
4. 백테스트로 적중률 검증
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

    pos_freq = {i: Counter() for i in range(6)}
    for r in recent:
        for pos, ball in enumerate(r['balls']):
            pos_freq[pos][ball] += 1

    first_freq = Counter(r['balls'][0] for r in recent)
    last_freq = Counter(r['balls'][5] for r in recent)
    pair_freq = Counter((r['balls'][0], r['balls'][5]) for r in recent)

    return {
        'all_freq': all_freq,
        'pos_freq': pos_freq,
        'first_freq': first_freq,
        'last_freq': last_freq,
        'pair_freq': pair_freq,
        'recent': recent
    }


def select_firstend_pairs(stats, seen_ratio=0.5, unseen_ratio=0.5):
    """
    firstend 쌍 선택 - 50% 학습데이터 + 50% 미출현 가능성

    1. 학습 데이터에서 나온 쌍 중 상위 50%
    2. 아직 나오지 않았지만 나올 가능성 높은 쌍 중 상위 50%
    """
    first_freq = stats['first_freq']
    last_freq = stats['last_freq']
    pair_freq = stats['pair_freq']
    recent = stats['recent']

    # 최근 3회 출현 번호 (페널티용)
    recent_3 = set()
    for r in recent[-3:]:
        recent_3.update(r['balls'])

    # 학습 데이터에서 실제로 나온 쌍들
    seen_pairs = set(pair_freq.keys())

    # 1. 학습 데이터에서 나온 쌍 (seen pairs)
    seen_list = []
    for (first, last), freq in pair_freq.items():
        score = 0
        score += freq * 10  # 실제 출현 빈도 가중치 높임
        score += first_freq.get(first, 0) * 2
        score += last_freq.get(last, 0) * 2

        # 최근 3회 페널티
        if first in recent_3:
            score -= 3
        if last in recent_3:
            score -= 3

        seen_list.append({'first': first, 'last': last, 'score': score, 'type': 'seen'})

    seen_list.sort(key=lambda x: -x['score'])

    # 2. 나오지 않았지만 나올 가능성 있는 쌍 (unseen pairs)
    unseen_list = []
    for first in range(1, 16):  # 첫수 범위: 1-15
        for last in range(first + 5, 46):  # 끝수: 첫수+5 ~ 45
            if (first, last) in seen_pairs:
                continue  # 이미 나온 쌍은 제외

            score = 0
            # 개별 번호의 빈도로 가능성 추정
            score += first_freq.get(first, 0) * 3
            score += last_freq.get(last, 0) * 3

            # 첫수 범위 보너스 (1-10이 가장 빈번)
            if 1 <= first <= 10:
                score += 5

            # 끝수 범위 보너스 (38-45가 가장 빈번)
            if 38 <= last <= 45:
                score += 8
            elif 35 <= last <= 37:
                score += 4

            # 최근 3회 페널티
            if first in recent_3:
                score -= 3
            if last in recent_3:
                score -= 3

            unseen_list.append({'first': first, 'last': last, 'score': score, 'type': 'unseen'})

    unseen_list.sort(key=lambda x: -x['score'])

    # 3. 각각 50%씩 선택
    n_seen = int(len(seen_list) * seen_ratio) if seen_list else 0
    n_unseen = int(len(unseen_list) * unseen_ratio) if unseen_list else 0

    # 최소 개수 보장
    n_seen = max(n_seen, min(30, len(seen_list)))
    n_unseen = max(n_unseen, min(50, len(unseen_list)))

    selected = seen_list[:n_seen] + unseen_list[:n_unseen]

    # 최종 점수순 정렬
    selected.sort(key=lambda x: -x['score'])

    return selected


def score_number(num, stats):
    all_freq = stats['all_freq']
    score = all_freq.get(num, 0) * 2

    if num in PRIMES:
        score += 5

    if 10 <= num <= 19:
        score += 3
    elif 20 <= num <= 29:
        score += 3
    elif 30 <= num <= 39:
        score += 4
    elif 40 <= num <= 45:
        score += 3

    hot_bits = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
    cold_bits = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    if num in hot_bits:
        score += 4
    if num in cold_bits:
        score -= 2

    return score


def score_all_numbers(stats):
    return {num: score_number(num, stats) for num in range(1, 46)}


def score_combination_bonus(balls):
    score = 0
    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
    elif 100 <= total_sum <= 180:
        score += 5

    consecutive = sum(1 for i in range(5) if balls[i+1] - balls[i] == 1)
    if consecutive == 1:
        score += 8
    elif consecutive == 0:
        score += 5
    elif consecutive == 2:
        score += 3

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

    prime_count = sum(1 for b in balls if b in PRIMES)
    if prime_count in [1, 2]:
        score += 10
    elif prime_count == 3:
        score += 5

    if 10 <= ball2 <= 19:
        score += 5
    if 10 <= ball3 <= 19:
        score += 5
    if 20 <= ball4 <= 29:
        score += 5
    if 30 <= ball5 <= 39:
        score += 8
    if 40 <= ball6 <= 45:
        score += 8

    return score


def generate_combos_for_round(data, round_idx, max_combos=500):
    past_data = data[:round_idx]
    if len(past_data) < 50:
        return []

    stats = get_recent_stats(past_data, 50)

    # 1단계: firstend 쌍 선택 (50% seen + 50% unseen)
    pairs = select_firstend_pairs(stats, seen_ratio=0.5, unseen_ratio=0.5)

    # 2단계: 번호 점수화
    number_scores = score_all_numbers(stats)
    sorted_numbers = sorted(range(1, 46), key=lambda x: -number_scores[x])

    # 3단계: 조합 생성
    all_combos = []

    # seen/unseen 타입별로 균형있게 사용 (seen 60%, unseen 40%)
    seen_pairs = [p for p in pairs if p['type'] == 'seen'][:50]
    unseen_pairs = [p for p in pairs if p['type'] == 'unseen'][:35]
    selected_pairs = seen_pairs + unseen_pairs

    for pair in selected_pairs:
        first = pair['first']
        last = pair['last']
        pair_score = pair['score']
        pair_type = pair['type']

        # 중간 번호 후보 (더 많이 선택)
        mid_candidates = [n for n in sorted_numbers if first < n < last][:20]

        for mid_combo in combinations(mid_candidates, 4):
            balls = sorted([first] + list(mid_combo) + [last])

            combo_score = pair_score
            for b in mid_combo:
                combo_score += number_scores[b]
            combo_score += score_combination_bonus(balls)

            # unseen 쌍에 약간의 보너스 (다양성)
            if pair_type == 'unseen':
                combo_score += 2

            all_combos.append({'balls': balls, 'score': combo_score, 'type': pair_type})

    # 중복 제거
    seen = set()
    unique = []
    for c in all_combos:
        key = tuple(c['balls'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique.sort(key=lambda x: -x['score'])
    return unique[:max_combos]


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

        combos = generate_combos_for_round(data, idx, combo_count)
        if not combos:
            continue

        match_counts = [len(set(c['balls']) & set(actual_balls)) for c in combos]
        best_match = max(match_counts) if match_counts else 0
        match_dist = Counter(match_counts)

        match_3plus = sum(1 for m in match_counts if m >= 3)
        match_4plus = sum(1 for m in match_counts if m >= 4)
        match_5plus = sum(1 for m in match_counts if m >= 5)

        # 첫수/끝수 적중 확인
        first_hit = any(c['balls'][0] == actual_balls[0] for c in combos)
        last_hit = any(c['balls'][5] == actual_balls[5] for c in combos)

        results.append({
            'round': target_round,
            'actual_balls': actual_balls,
            'best_match': best_match,
            'top1_match': match_counts[0] if match_counts else 0,
            'match_3plus': match_3plus,
            'match_4plus': match_4plus,
            'match_5plus': match_5plus,
            'first_hit': first_hit,
            'last_hit': last_hit,
            'total_combos': len(combos)
        })

        progress = (idx - start_idx + 1) / total * 100
        if (idx - start_idx + 1) % 20 == 0 or idx == end_idx - 1:
            print(f"[{progress:5.1f}%] {target_round}회: 최고={best_match}개, "
                  f"3+={match_3plus}, 4+={match_4plus}, 5+={match_5plus}, "
                  f"첫수={first_hit}, 끝수={last_hit}")

    return results


def analyze_results(results):
    print("\n" + "=" * 70)
    print("백테스트 결과 (v2 - firstend + 인사이트 점수화)")
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

    # 첫수/끝수 적중률
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
    print("백테스트 v2 - firstend + 인사이트 점수화")
    print("=" * 70)

    data = load_data()
    print(f"데이터: {len(data)}회차")

    results = run_backtest(data, args.start, args.count, args.max_rounds)

    if results:
        analyze_results(results)


if __name__ == "__main__":
    main()
