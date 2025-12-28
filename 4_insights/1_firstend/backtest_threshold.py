"""
다양한 선택 비율에 따른 적중률 테스트
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"


def load_all_data() -> list:
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            results.append({
                'round': int(row['round']),
                'first': balls[0],
                'end': balls[-1]
            })
    return results


def build_distributions(data: list) -> tuple:
    first_dist = Counter(r['first'] for r in data)
    end_dist = Counter(r['end'] for r in data)
    pair_dist = Counter((r['first'], r['end']) for r in data)
    return first_dist, end_dist, pair_dist


def calculate_scores(first_dist, end_dist, pair_dist, recent, penalty=5):
    scores = []
    for first in range(1, 31):
        for end in range(max(first + 7, 17), 46):
            total = first_dist.get(first, 0) + end_dist.get(end, 0) + pair_dist.get((first, end), 0)
            if first in recent:
                total -= penalty
            if end in recent:
                total -= penalty
            scores.append({'first': first, 'end': end, 'total': total})
    scores.sort(key=lambda x: x['total'], reverse=True)
    return scores


def select_top_percent(scores: list, percent: float) -> set:
    total_sum = sum(s['total'] for s in scores if s['total'] > 0)
    selected = set()
    cumsum = 0
    for s in scores:
        if s['total'] <= 0:
            continue
        cumsum += s['total']
        selected.add((s['first'], s['end']))
        if cumsum >= total_sum * percent:
            break
    return selected


def backtest_percent(percent: float, all_data: list, start_idx: int):
    hits = 0
    total = 0
    selected_counts = []

    for i in range(start_idx, len(all_data)):
        target = all_data[i]
        past_data = all_data[:i]

        if len(past_data) < 50:
            continue

        first_dist, end_dist, pair_dist = build_distributions(past_data)
        recent = set()
        for r in past_data[-3:]:
            recent.add(r['first'])
            recent.add(r['end'])

        scores = calculate_scores(first_dist, end_dist, pair_dist, recent)
        selected = select_top_percent(scores, percent)
        selected_counts.append(len(selected))

        if (target['first'], target['end']) in selected:
            hits += 1
        total += 1

    avg_selected = sum(selected_counts) / len(selected_counts) if selected_counts else 0
    return hits, total, avg_selected


def main():
    all_data = load_all_data()
    start_idx = 50  # 826 + 50 = 876회차부터

    print("=== 선택 비율별 적중률 테스트 ===\n")
    print(f"{'비율':>6} {'적중':>6} {'총':>6} {'적중률':>8} {'평균선택수':>10}")
    print("-" * 45)

    results = []
    for percent in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        hits, total, avg_selected = backtest_percent(percent, all_data, start_idx)
        hit_rate = hits / total * 100
        print(f"{percent*100:>5.0f}% {hits:>6} {total:>6} {hit_rate:>7.1f}% {avg_selected:>10.0f}")
        results.append({
            'percent': percent,
            'hits': hits,
            'total': total,
            'hit_rate': hit_rate,
            'avg_selected': avg_selected
        })

    print("\n" + "=" * 45)

    # 최적 비율 추천
    print("\n추천:")
    for r in results:
        if r['hit_rate'] >= 90:
            print(f"  {r['percent']*100:.0f}%: 적중률 {r['hit_rate']:.1f}%, 평균 {r['avg_selected']:.0f}개 선택")


if __name__ == "__main__":
    main()
