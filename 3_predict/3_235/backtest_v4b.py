"""
백테스트 v4b - 순차적 50/50 + 혼합 전략 결합

v4의 순차적 50/50 전략과 v3의 혼합 전략(Top24+Rest21)을 결합

전략:
1. 각 포지션별로 50% seen + 50% unseen으로 후보 선택
2. Top24/Rest21 패턴에 따라 조합 생성
3. 순차적 의존성 유지 (ball1 < ball2 < ... < ball6)
"""

import csv
from pathlib import Path
from collections import Counter
from itertools import combinations
import statistics

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

# 포지션별 최빈 구간 정의
OPTIMAL_RANGES = {
    0: (1, 10),    # ball1 (firstend - 첫수)
    1: (10, 19),   # ball2 (second - 48.3%)
    2: (10, 19),   # ball3 (third - 45.4%)
    3: (20, 29),   # ball4 (fourth - 42.7%)
    4: (30, 39),   # ball5 (fifth - 54.6%)
    5: (40, 45),   # ball6 (lastnum - 57.8%)
}

# 핫/콜드 비트 (12_onehot)
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}


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

    # Top24 계산 (빈도순 상위 24개)
    sorted_by_freq = sorted(range(1, 46), key=lambda x: -all_freq.get(x, 0))
    top24 = set(sorted_by_freq[:24])
    rest21 = set(sorted_by_freq[24:])

    return {
        'all_freq': all_freq,
        'pos_freq': pos_freq,
        'top24': top24,
        'rest21': rest21,
        'recent': recent
    }


def select_position_candidates_5050(pos, stats):
    """
    각 포지션별 후보 선택 (50% seen + 50% unseen)
    모든 가능한 번호를 반환 (순서 제약 없이)
    """
    pos_freq = stats['pos_freq'][pos]
    all_freq = stats['all_freq']
    recent = stats['recent']
    optimal_min, optimal_max = OPTIMAL_RANGES[pos]

    # 최근 3회 출현 번호
    recent_3 = set()
    for r in recent[-3:]:
        recent_3.update(r['balls'])

    # seen: 해당 포지션에서 실제로 나온 번호들
    seen_numbers = set(pos_freq.keys())

    # 1. seen list
    seen_list = []
    for num in seen_numbers:
        freq = pos_freq[num]
        score = freq * 10

        # 최빈 구간 보너스
        if optimal_min <= num <= optimal_max:
            score += 15

        # 핫/콜드 비트
        if num in HOT_BITS:
            score += 5
        if num in COLD_BITS:
            score -= 3

        # 최근 3회 페널티
        if num in recent_3:
            score -= 5

        # 소수 보너스
        if num in PRIMES:
            score += 3

        seen_list.append({'num': num, 'score': score, 'type': 'seen'})

    seen_list.sort(key=lambda x: -x['score'])

    # 2. unseen list
    unseen_list = []
    for num in range(1, 46):
        if num in seen_numbers:
            continue

        score = 0

        # 최빈 구간 보너스
        if optimal_min <= num <= optimal_max:
            score += 20

        # 전체 빈도 참조
        score += all_freq.get(num, 0) * 2

        # 핫/콜드 비트
        if num in HOT_BITS:
            score += 5
        if num in COLD_BITS:
            score -= 3

        # 최근 3회 페널티
        if num in recent_3:
            score -= 5

        # 소수 보너스
        if num in PRIMES:
            score += 3

        unseen_list.append({'num': num, 'score': score, 'type': 'unseen'})

    unseen_list.sort(key=lambda x: -x['score'])

    # 50/50 선택
    n_seen = max(int(len(seen_list) * 0.5), min(12, len(seen_list)))
    n_unseen = max(int(len(unseen_list) * 0.5), min(8, len(unseen_list)))

    selected = seen_list[:n_seen] + unseen_list[:n_unseen]
    return {c['num']: c['score'] for c in selected}


def score_combination_bonus(balls):
    """조합 전체에 대한 추가 점수"""
    score = 0
    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # 합계 점수 (121-160이 48.5%)
    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
    elif 100 <= total_sum <= 180:
        score += 5

    # 연속수 점수 (1쌍 40.9%, 0쌍 47.8%)
    consecutive = sum(1 for i in range(5) if balls[i+1] - balls[i] == 1)
    if consecutive == 1:
        score += 8
    elif consecutive == 0:
        score += 5
    elif consecutive == 2:
        score += 3

    # 구간 분포 (4개 구간 51.2%)
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

    # 소수 개수 (1-2개 66.7%)
    prime_count = sum(1 for b in balls if b in PRIMES)
    if prime_count in [1, 2]:
        score += 10
    elif prime_count == 3:
        score += 5

    # 포지션별 최빈 구간 보너스
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


def generate_combos_hybrid(data, round_idx, max_combos=500):
    """
    순차적 50/50 + 혼합 전략 결합

    1. 각 포지션별로 50/50 후보 선택
    2. Top24/Rest21 패턴에 따라 후보 분류
    3. 패턴별 조합 생성
    """
    past_data = data[:round_idx]
    if len(past_data) < 50:
        return []

    stats = get_recent_stats(past_data, 50)
    top24 = stats['top24']
    rest21 = stats['rest21']

    # 각 포지션별 50/50 후보 선택
    pos_candidates = {}
    pos_scores = {}
    for pos in range(6):
        candidates = select_position_candidates_5050(pos, stats)
        pos_candidates[pos] = list(candidates.keys())
        pos_scores[pos] = candidates

    # 각 포지션 후보를 Top24/Rest21로 분류
    pos_top = {}
    pos_rest = {}
    for pos in range(6):
        pos_top[pos] = sorted([n for n in pos_candidates[pos] if n in top24])
        pos_rest[pos] = sorted([n for n in pos_candidates[pos] if n in rest21])

    # 혼합 패턴 정의 (Top24에서 몇 개, Rest21에서 몇 개)
    # 분석 결과: Top24에서 평균 3.11개, Rest21에서 2.89개
    patterns = []

    # Top4 패턴 (4개 포지션 Top24, 2개 포지션 Rest21)
    for top_positions in combinations(range(6), 4):
        rest_positions = [i for i in range(6) if i not in top_positions]
        patterns.append({
            'top_positions': top_positions,
            'rest_positions': rest_positions,
            'type': 'top4'
        })

    # Top3 패턴 (3개 포지션 Top24, 3개 포지션 Rest21)
    for top_positions in combinations(range(6), 3):
        rest_positions = [i for i in range(6) if i not in top_positions]
        patterns.append({
            'top_positions': top_positions,
            'rest_positions': rest_positions,
            'type': 'top3'
        })

    all_combos = []

    for pattern in patterns:
        top_positions = pattern['top_positions']
        rest_positions = pattern['rest_positions']

        # 각 포지션에서 후보 선택
        position_pools = []
        for pos in range(6):
            if pos in top_positions:
                pool = pos_top[pos][:10] if pos_top[pos] else pos_candidates[pos][:10]
            else:
                pool = pos_rest[pos][:8] if pos_rest[pos] else pos_candidates[pos][:8]
            position_pools.append(pool)

        # 조합 생성 (순차 조건: b1 < b2 < ... < b6)
        for b1 in position_pools[0]:
            for b2 in position_pools[1]:
                if b2 <= b1:
                    continue
                for b3 in position_pools[2]:
                    if b3 <= b2:
                        continue
                    for b4 in position_pools[3]:
                        if b4 <= b3:
                            continue
                        for b5 in position_pools[4]:
                            if b5 <= b4:
                                continue
                            for b6 in position_pools[5]:
                                if b6 <= b5:
                                    continue

                                balls = [b1, b2, b3, b4, b5, b6]

                                # 점수 계산
                                combo_score = sum(pos_scores[i].get(balls[i], 0) for i in range(6))
                                combo_score += score_combination_bonus(balls)

                                # 패턴 보너스
                                if pattern['type'] == 'top4':
                                    combo_score += 5
                                elif pattern['type'] == 'top3':
                                    combo_score += 3

                                all_combos.append({
                                    'balls': balls,
                                    'score': combo_score,
                                    'pattern': pattern['type']
                                })

    # 중복 제거
    seen = set()
    unique = []
    for c in all_combos:
        key = tuple(c['balls'])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # 점수순 정렬
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

        combos = generate_combos_hybrid(data, idx, combo_count)
        if not combos:
            continue

        match_counts = [len(set(c['balls']) & set(actual_balls)) for c in combos]
        best_match = max(match_counts) if match_counts else 0

        match_3plus = sum(1 for m in match_counts if m >= 3)
        match_4plus = sum(1 for m in match_counts if m >= 4)
        match_5plus = sum(1 for m in match_counts if m >= 5)

        # 포지션별 적중 확인
        pos_hits = [False] * 6
        for c in combos:
            for i in range(6):
                if c['balls'][i] == actual_balls[i]:
                    pos_hits[i] = True

        # Top24 적중 개수
        stats = get_recent_stats(data[:idx], 50)
        top24_hit = sum(1 for b in actual_balls if b in stats['top24'])

        results.append({
            'round': target_round,
            'actual_balls': actual_balls,
            'best_match': best_match,
            'top1_match': match_counts[0] if match_counts else 0,
            'match_3plus': match_3plus,
            'match_4plus': match_4plus,
            'match_5plus': match_5plus,
            'pos_hits': pos_hits,
            'top24_hit': top24_hit,
            'total_combos': len(combos)
        })

        progress = (idx - start_idx + 1) / total * 100
        if (idx - start_idx + 1) % 20 == 0 or idx == end_idx - 1:
            pos_hit_str = ''.join(['O' if h else 'X' for h in pos_hits])
            print(f"[{progress:5.1f}%] {target_round}회: Top24={top24_hit}, 최고={best_match}개, "
                  f"3+={match_3plus}, 4+={match_4plus}, 포지션={pos_hit_str}")

    return results


def analyze_results(results):
    print("\n" + "=" * 70)
    print("백테스트 결과 (v4b - 순차적 50/50 + 혼합 전략)")
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

    r3 = sum(1 for r in results if r['best_match'] >= 3)
    r4 = sum(1 for r in results if r['best_match'] >= 4)
    r5 = sum(1 for r in results if r['best_match'] >= 5)
    r6 = sum(1 for r in results if r['best_match'] == 6)

    print(f"\n[N개 이상 적중 비율]")
    print(f"  3개+: {r3}/{total} ({r3/total*100:.1f}%)")
    print(f"  4개+: {r4}/{total} ({r4/total*100:.1f}%)")
    print(f"  5개+: {r5}/{total} ({r5/total*100:.1f}%)")
    print(f"  6개:  {r6}/{total} ({r6/total*100:.1f}%)")

    # 포지션별 적중률
    pos_hit_counts = [0] * 6
    for r in results:
        for i in range(6):
            if r['pos_hits'][i]:
                pos_hit_counts[i] += 1

    print(f"\n[포지션별 적중률]")
    pos_names = ['ball1(첫수)', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6(끝수)']
    for i in range(6):
        pct = pos_hit_counts[i] / total * 100
        print(f"  {pos_names[i]}: {pos_hit_counts[i]}/{total} ({pct:.1f}%)")

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
        'pos_hit_pcts': [c / total * 100 for c in pos_hit_counts]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=900)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--max-rounds', type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("백테스트 v4b - 순차적 50/50 + 혼합 전략")
    print("=" * 70)
    print("\n전략:")
    print("  1. 각 포지션별 50% seen + 50% unseen 후보 선택")
    print("  2. Top24/Rest21 혼합 패턴 적용")
    print("  3. 순차 조건 유지 (ball1 < ball2 < ... < ball6)")

    data = load_data()
    print(f"\n데이터: {len(data)}회차")

    results = run_backtest(data, args.start, args.count, args.max_rounds)

    if results:
        analyze_results(results)


if __name__ == "__main__":
    main()
