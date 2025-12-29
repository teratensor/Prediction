"""
6개 적중에 필요한 최소 후보 수 분석
"""

import csv
from pathlib import Path
from analyze_no_6match import (
    load_all_data, get_ord2_candidates, get_ord3_candidates,
    get_ord4_candidates, get_ord5_candidates
)

BASE_DIR = Path(__file__).parent.parent


def analyze_min_candidates(data, target_round):
    """6개 적중에 필요한 각 단계별 최소 후보 수 계산"""

    actual_data = next((d for d in data if d['round'] == target_round), None)
    if not actual_data:
        return None

    actual = actual_data['balls']
    a1, a2, a3, a4, a5, a6 = actual

    train_data = [d for d in data if d['round'] < target_round]
    winning_sets = [d['balls'] for d in train_data]

    result = {
        'round': target_round,
        'actual': actual,
        'needed': {}
    }

    # Second: 실제 ord2의 순위
    ord2_all = get_ord2_candidates(a1, a6, max_candidates=100)
    for i, c in enumerate(ord2_all):
        if c['ord2'] == a2:
            result['needed']['second'] = i + 1
            break
    else:
        result['needed']['second'] = -1  # 범위 밖

    # Third
    ord3_all = get_ord3_candidates(a1, a2, a6, max_candidates=100)
    for i, c in enumerate(ord3_all):
        if c['ord3'] == a3:
            result['needed']['third'] = i + 1
            break
    else:
        result['needed']['third'] = -1

    # Fourth
    ord4_all = get_ord4_candidates(a1, a2, a3, a6, max_candidates=100)
    for i, c in enumerate(ord4_all):
        if c['ord4'] == a4:
            result['needed']['fourth'] = i + 1
            break
    else:
        result['needed']['fourth'] = -1

    # Fifth
    ord5_all = get_ord5_candidates(a1, a2, a3, a4, a6, winning_sets, max_candidates=100)
    non_dup = [c for c in ord5_all if not c['is_duplicate']]
    for i, c in enumerate(non_dup):
        if c['ord5'] == a5:
            result['needed']['fifth'] = i + 1
            break
    else:
        result['needed']['fifth'] = -1

    return result


def main():
    print("=" * 80)
    print("6개 적중에 필요한 최소 후보 수 분석")
    print("=" * 80)

    data = load_all_data()

    start_round, end_round = 900, 1100

    # 각 단계별 필요 순위 수집
    needed_ranks = {
        'second': [],
        'third': [],
        'fourth': [],
        'fifth': []
    }

    for target_round in range(start_round, end_round + 1):
        result = analyze_min_candidates(data, target_round)
        if result is None:
            continue

        for stage in ['second', 'third', 'fourth', 'fifth']:
            rank = result['needed'].get(stage, -1)
            if rank > 0:
                needed_ranks[stage].append(rank)

    # 통계
    print("\n<단계별 실제 당첨번호 순위 분포>")
    print("-" * 80)

    for stage in ['second', 'third', 'fourth', 'fifth']:
        ranks = needed_ranks[stage]
        if not ranks:
            continue

        avg = sum(ranks) / len(ranks)
        top2_rate = sum(1 for r in ranks if r <= 2) / len(ranks) * 100
        top3_rate = sum(1 for r in ranks if r <= 3) / len(ranks) * 100
        top5_rate = sum(1 for r in ranks if r <= 5) / len(ranks) * 100
        top10_rate = sum(1 for r in ranks if r <= 10) / len(ranks) * 100

        print(f"\n[{stage}]")
        print(f"  평균 순위: {avg:.1f}")
        print(f"  Top 2 포함률: {top2_rate:.1f}%")
        print(f"  Top 3 포함률: {top3_rate:.1f}%")
        print(f"  Top 5 포함률: {top5_rate:.1f}%")
        print(f"  Top 10 포함률: {top10_rate:.1f}%")

    # 시뮬레이션: 후보 수 변경 시 6개 적중 가능 비율
    print("\n" + "=" * 80)
    print("<후보 수 변경 시 6개 적중 가능 회차 시뮬레이션>")
    print("=" * 80)

    for max_cand in [2, 3, 5, 10]:
        passed = 0
        total = 0

        for target_round in range(start_round, end_round + 1):
            result = analyze_min_candidates(data, target_round)
            if result is None:
                continue

            total += 1

            # 모든 단계에서 max_cand 이내에 있는지
            all_within = True
            for stage in ['second', 'third', 'fourth', 'fifth']:
                rank = result['needed'].get(stage, -1)
                if rank <= 0 or rank > max_cand:
                    all_within = False
                    break

            if all_within:
                passed += 1

        print(f"  max_candidates={max_cand:2d}: {passed:3d}회 ({passed/total*100:5.1f}%) 6개 적중 가능")


if __name__ == "__main__":
    main()
