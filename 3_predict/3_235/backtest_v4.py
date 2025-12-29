"""
백테스트 v4 - 순차적 50/50 전략

전략:
1. ball1 선택: firstend 인사이트 (50% seen + 50% unseen)
2. ball2 선택: ball1 기준, second 인사이트 (50% seen + 50% unseen)
3. ball3 선택: ball2 기준, third 인사이트 (50% seen + 50% unseen)
4. ball4 선택: ball3 기준, fourth 인사이트 (50% seen + 50% unseen)
5. ball5 선택: ball4 기준, fifth 인사이트 (50% seen + 50% unseen)
6. ball6 선택: ball5 기준, lastnum 인사이트 (50% seen + 50% unseen)
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

    # 연속 번호 쌍 (position i → position i+1)
    pos_pairs = {i: Counter() for i in range(5)}
    for r in recent:
        for i in range(5):
            pos_pairs[i][(r['balls'][i], r['balls'][i+1])] += 1

    return {
        'all_freq': all_freq,
        'pos_freq': pos_freq,
        'pos_pairs': pos_pairs,
        'recent': recent
    }


def select_ball1_candidates(stats, seen_ratio=0.5, unseen_ratio=0.5):
    """
    ball1 (첫수) 선택 - firstend 인사이트 적용
    최빈 구간: 1-10
    """
    pos_freq = stats['pos_freq'][0]
    recent = stats['recent']
    optimal_min, optimal_max = OPTIMAL_RANGES[0]

    # 최근 3회 출현 번호 (페널티용)
    recent_3 = set()
    for r in recent[-3:]:
        recent_3.update(r['balls'])

    # seen: 실제로 첫수로 나온 번호들
    seen_numbers = set(pos_freq.keys())

    # 1. seen list (학습 데이터에서 나온 번호)
    seen_list = []
    for num, freq in pos_freq.items():
        score = freq * 10
        if optimal_min <= num <= optimal_max:
            score += 15  # 최빈 구간 보너스
        if num in HOT_BITS:
            score += 5
        if num in COLD_BITS:
            score -= 3
        if num in recent_3:
            score -= 5

        seen_list.append({'num': num, 'score': score, 'type': 'seen'})

    seen_list.sort(key=lambda x: -x['score'])

    # 2. unseen list (아직 안 나왔지만 나올 가능성)
    unseen_list = []
    for num in range(1, 46):
        if num in seen_numbers:
            continue

        score = 0
        # 최빈 구간 보너스
        if optimal_min <= num <= optimal_max:
            score += 20
        # 전체 빈도 참조
        score += stats['all_freq'].get(num, 0) * 2
        if num in HOT_BITS:
            score += 5
        if num in COLD_BITS:
            score -= 3
        if num in recent_3:
            score -= 5

        unseen_list.append({'num': num, 'score': score, 'type': 'unseen'})

    unseen_list.sort(key=lambda x: -x['score'])

    # 50/50 선택
    n_seen = max(int(len(seen_list) * seen_ratio), min(10, len(seen_list)))
    n_unseen = max(int(len(unseen_list) * unseen_ratio), min(5, len(unseen_list)))

    selected = seen_list[:n_seen] + unseen_list[:n_unseen]
    selected.sort(key=lambda x: -x['score'])

    return selected


def select_next_ball_candidates(pos, prev_ball, stats, seen_ratio=0.5, unseen_ratio=0.5):
    """
    ball2-6 선택 - 이전 번호보다 큰 번호 중에서 선택
    pos: 0-5 (ball1-6에 해당, 실제로는 1-5로 호출됨)
    prev_ball: 이전 포지션에서 선택된 번호
    """
    pos_freq = stats['pos_freq'][pos]
    recent = stats['recent']
    optimal_min, optimal_max = OPTIMAL_RANGES[pos]

    # 최근 3회 출현 번호
    recent_3 = set()
    for r in recent[-3:]:
        recent_3.update(r['balls'])

    # seen: 해당 포지션에서 실제로 나온 번호들 (prev_ball보다 큰 것)
    seen_numbers = set(num for num in pos_freq.keys() if num > prev_ball)

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

    # 2. unseen list (해당 포지션에서 안 나왔지만 나올 가능성)
    unseen_list = []
    for num in range(prev_ball + 1, 46):
        if num in seen_numbers:
            continue

        score = 0

        # 최빈 구간 보너스
        if optimal_min <= num <= optimal_max:
            score += 20

        # 전체 빈도 참조
        score += stats['all_freq'].get(num, 0) * 2

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
    n_seen = max(int(len(seen_list) * seen_ratio), min(8, len(seen_list)))
    n_unseen = max(int(len(unseen_list) * unseen_ratio), min(4, len(unseen_list)))

    selected = seen_list[:n_seen] + unseen_list[:n_unseen]
    selected.sort(key=lambda x: -x['score'])

    return selected


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


def generate_combos_sequential(data, round_idx, max_combos=500):
    """순차적 50/50 전략으로 조합 생성 (개선 버전)"""
    past_data = data[:round_idx]
    if len(past_data) < 50:
        return []

    stats = get_recent_stats(past_data, 50)

    # 1단계: ball1 후보 선택 (더 많은 후보 사용)
    ball1_candidates = select_ball1_candidates(stats)
    ball1_nums = [c['num'] for c in ball1_candidates[:20]]

    all_combos = []

    for b1 in ball1_nums:
        b1_score = next((c['score'] for c in ball1_candidates if c['num'] == b1), 0)

        # 2단계: ball2 후보 선택 (b1보다 큰 것)
        ball2_candidates = select_next_ball_candidates(1, b1, stats)
        ball2_nums = [c['num'] for c in ball2_candidates[:15]]

        for b2 in ball2_nums:
            b2_score = next((c['score'] for c in ball2_candidates if c['num'] == b2), 0)

            # 3단계: ball3 후보 선택 (b2보다 큰 것)
            ball3_candidates = select_next_ball_candidates(2, b2, stats)
            ball3_nums = [c['num'] for c in ball3_candidates[:12]]

            for b3 in ball3_nums:
                b3_score = next((c['score'] for c in ball3_candidates if c['num'] == b3), 0)

                # 4단계: ball4 후보 선택 (b3보다 큰 것)
                ball4_candidates = select_next_ball_candidates(3, b3, stats)
                ball4_nums = [c['num'] for c in ball4_candidates[:10]]

                for b4 in ball4_nums:
                    b4_score = next((c['score'] for c in ball4_candidates if c['num'] == b4), 0)

                    # 5단계: ball5 후보 선택 (b4보다 큰 것)
                    ball5_candidates = select_next_ball_candidates(4, b4, stats)
                    ball5_nums = [c['num'] for c in ball5_candidates[:8]]

                    for b5 in ball5_nums:
                        b5_score = next((c['score'] for c in ball5_candidates if c['num'] == b5), 0)

                        # 6단계: ball6 후보 선택 (b5보다 큰 것)
                        ball6_candidates = select_next_ball_candidates(5, b5, stats)
                        ball6_nums = [c['num'] for c in ball6_candidates[:6]]

                        for b6 in ball6_nums:
                            b6_score = next((c['score'] for c in ball6_candidates if c['num'] == b6), 0)

                            balls = [b1, b2, b3, b4, b5, b6]

                            # 총 점수 계산
                            combo_score = b1_score + b2_score + b3_score + b4_score + b5_score + b6_score
                            combo_score += score_combination_bonus(balls)

                            all_combos.append({
                                'balls': balls,
                                'score': combo_score
                            })

    # 중복 제거 (이론적으로 없어야 하지만 안전을 위해)
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

        combos = generate_combos_sequential(data, idx, combo_count)
        if not combos:
            continue

        match_counts = [len(set(c['balls']) & set(actual_balls)) for c in combos]
        best_match = max(match_counts) if match_counts else 0
        match_dist = Counter(match_counts)

        match_3plus = sum(1 for m in match_counts if m >= 3)
        match_4plus = sum(1 for m in match_counts if m >= 4)
        match_5plus = sum(1 for m in match_counts if m >= 5)

        # 포지션별 적중 확인
        pos_hits = [False] * 6
        for c in combos:
            for i in range(6):
                if c['balls'][i] == actual_balls[i]:
                    pos_hits[i] = True

        results.append({
            'round': target_round,
            'actual_balls': actual_balls,
            'best_match': best_match,
            'top1_match': match_counts[0] if match_counts else 0,
            'match_3plus': match_3plus,
            'match_4plus': match_4plus,
            'match_5plus': match_5plus,
            'pos_hits': pos_hits,
            'total_combos': len(combos)
        })

        progress = (idx - start_idx + 1) / total * 100
        if (idx - start_idx + 1) % 20 == 0 or idx == end_idx - 1:
            pos_hit_str = ''.join(['O' if h else 'X' for h in pos_hits])
            print(f"[{progress:5.1f}%] {target_round}회: 최고={best_match}개, "
                  f"3+={match_3plus}, 4+={match_4plus}, 5+={match_5plus}, "
                  f"포지션={pos_hit_str}")

    return results


def analyze_results(results):
    print("\n" + "=" * 70)
    print("백테스트 결과 (v4 - 순차적 50/50 전략)")
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
    print("백테스트 v4 - 순차적 50/50 전략")
    print("=" * 70)
    print("\n전략:")
    print("  ball1 → ball2 → ball3 → ball4 → ball5 → ball6")
    print("  각 포지션: 50% seen + 50% unseen, 이전 번호보다 큰 것만 선택")

    data = load_data()
    print(f"\n데이터: {len(data)}회차")

    results = run_backtest(data, args.start, args.count, args.max_rounds)

    if results:
        analyze_results(results)


if __name__ == "__main__":
    main()
