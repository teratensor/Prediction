"""
백테스트 - 실제 ord1, ord6 고정 + max_candidates=15

각 회차마다 실제 당첨번호의 첫수/끝수를 알고 있다고 가정하고
나머지 ord2,3,4,5를 예측하여 6개 적중 가능 여부 확인
"""

import csv
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"

# 상수
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

def get_range(num):
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4

def load_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': tuple(sorted([int(row[f'ball{i}']) for i in range(1, 7)])),
            })
    return sorted(results, key=lambda x: x['round'])


# ========== Second ==========
BALL2_STATS = {7: 25, 16: 23, 11: 23, 6: 23, 9: 23, 12: 21, 14: 20, 18: 20, 13: 20, 10: 17, 4: 17, 8: 16, 15: 16, 5: 13, 21: 12, 17: 12, 19: 11, 3: 9, 26: 8, 20: 8, 23: 8, 24: 6, 25: 5, 2: 5}
BALL1_BALL2_STATS = {1: {'avg': 7.6, 'mode': 9}, 2: {'avg': 9.1, 'mode': 6}, 3: {'avg': 9.2, 'mode': 6}, 4: {'avg': 10.4, 'mode': 7}, 5: {'avg': 11.3, 'mode': 12}, 6: {'avg': 12.4, 'mode': 7}, 7: {'avg': 12.4, 'mode': 9}, 8: {'avg': 14.3, 'mode': 11}, 9: {'avg': 14.4, 'mode': 14}, 10: {'avg': 16.5, 'mode': 16}}
GAP12_STATS = {1: 47, 2: 38, 3: 40, 4: 50, 5: 32, 6: 33, 7: 18, 8: 26, 9: 13, 10: 15}
ORD2_BIT_FREQ = {9: 28, 10: 26, 5: 23, 17: 22, 6: 21, 7: 20, 11: 20, 12: 20, 15: 20, 19: 19, 13: 18, 14: 17, 18: 15, 20: 14}

def get_ord2_candidates(ord1, ord6, max_candidates=15):
    min_ord2 = ord1 + 1
    max_ord2 = ord6 - 4
    if min_ord2 > max_ord2:
        return []

    candidates = []
    for ord2 in range(min_ord2, max_ord2 + 1):
        score = 0
        if ord2 in BALL2_STATS:
            score += BALL2_STATS[ord2] * 2
        if ord1 in BALL1_BALL2_STATS:
            stats = BALL1_BALL2_STATS[ord1]
            if ord2 == stats['mode']:
                score += 15
            if abs(ord2 - stats['avg']) <= 2:
                score += 10
        gap = ord2 - ord1
        if gap in GAP12_STATS:
            score += GAP12_STATS[gap] // 2
        if 10 <= ord2 <= 19:
            score += 10
        if ord2 in ORD2_BIT_FREQ:
            score += ORD2_BIT_FREQ[ord2]
        if ord2 in HOT_BITS:
            score += 10
        elif ord2 in COLD_BITS:
            score -= 5
        candidates.append({'ord2': ord2, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


# ========== Third ==========
BALL3_STATS = {17: 25, 18: 23, 19: 22, 12: 21, 14: 21, 20: 20, 16: 20, 13: 19, 21: 18, 22: 18, 15: 17, 11: 16, 23: 15, 24: 14, 10: 13, 25: 12, 9: 11, 26: 10}
GAP23_STATS = {1: 55, 2: 48, 3: 43, 4: 37, 5: 32, 6: 28, 7: 22, 8: 19, 9: 15, 10: 13}
ORD3_BIT_FREQ = {17: 23, 18: 22, 19: 21, 12: 20, 14: 20, 20: 19, 16: 19, 13: 18, 21: 17, 22: 17, 15: 16, 11: 15}

def get_ord3_candidates(ord1, ord2, ord6, max_candidates=15):
    min_ord3 = ord2 + 1
    max_ord3 = ord6 - 3
    if min_ord3 > max_ord3:
        return []

    candidates = []
    for ord3 in range(min_ord3, max_ord3 + 1):
        score = 0
        if ord3 in BALL3_STATS:
            score += BALL3_STATS[ord3] * 2
        gap = ord3 - ord2
        if gap in GAP23_STATS:
            score += GAP23_STATS[gap] // 2
        if 10 <= ord3 <= 19:
            score += 10
        if ord3 in ORD3_BIT_FREQ:
            score += ORD3_BIT_FREQ[ord3]
        if ord3 in HOT_BITS:
            score += 10
        elif ord3 in COLD_BITS:
            score -= 5
        candidates.append({'ord3': ord3, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


# ========== Fourth ==========
BALL4_STATS = {30: 26, 26: 19, 29: 18, 35: 18, 27: 18, 23: 18, 33: 17, 28: 16, 22: 16, 32: 16, 31: 16, 21: 15, 24: 15, 25: 14, 19: 14, 20: 13}
GAP34_STATS = {1: 46, 2: 54, 3: 44, 4: 37, 5: 24, 6: 30, 7: 25, 8: 22, 9: 18, 10: 18}
ORD4_BIT_FREQ = {27: 23, 28: 22, 25: 20, 24: 19, 29: 19, 31: 18, 21: 17, 17: 17, 32: 17, 33: 17, 23: 16}
UNEXPECTED_NUMBERS = {14, 37, 15, 36, 13, 34, 17, 40, 39, 41, 11, 12, 43, 42, 8}

def get_ord4_candidates(ord1, ord2, ord3, ord6, max_candidates=15):
    min_ord4 = ord3 + 1
    max_ord4 = ord6 - 1
    if min_ord4 > max_ord4:
        return []

    candidates = []
    for ord4 in range(min_ord4, max_ord4 + 1):
        score = 0
        if ord4 in BALL4_STATS:
            score += BALL4_STATS[ord4]
        if ord4 in UNEXPECTED_NUMBERS:
            score += 15
        if 20 <= ord4 <= 29:
            score += 8
        elif 30 <= ord4 <= 39:
            score += 6
        gap = ord4 - ord3
        if gap in GAP34_STATS:
            score += GAP34_STATS[gap] // 4
        if ord4 in ORD4_BIT_FREQ:
            score += ORD4_BIT_FREQ[ord4] // 2
        if ord4 in COLD_BITS:
            score += 12
        elif ord4 in HOT_BITS:
            score -= 3
        candidates.append({'ord4': ord4, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


# ========== Fifth ==========
BALL5_STATS = {33: 26, 34: 24, 38: 23, 37: 22, 39: 21, 31: 20, 36: 20, 35: 19, 32: 18, 28: 18, 40: 17, 29: 17, 30: 16, 27: 15, 42: 14, 26: 13, 41: 13}
GAP45_STATS = {1: 56, 2: 51, 3: 42, 4: 35, 5: 33, 6: 29, 7: 26, 8: 19, 9: 18, 10: 14}
SUM_OPTIMAL_MIN, SUM_OPTIMAL_MAX = 121, 160
SUM_GOOD_MIN, SUM_GOOD_MAX = 100, 170

def get_ord5_candidates(ord1, ord2, ord3, ord4, ord6, winning_sets, max_candidates=15):
    min_ord5 = ord4 + 1
    max_ord5 = ord6 - 1
    if min_ord5 > max_ord5:
        return []

    current_sum = ord1 + ord2 + ord3 + ord4 + ord6
    candidates = []

    for ord5 in range(min_ord5, max_ord5 + 1):
        combo = tuple(sorted([ord1, ord2, ord3, ord4, ord5, ord6]))

        # 5개/6개 중복 체크
        is_duplicate = False
        for winning in winning_sets:
            if len(set(combo) & set(winning)) >= 5:
                is_duplicate = True
                break

        score = 0
        total_sum = current_sum + ord5

        if SUM_OPTIMAL_MIN <= total_sum <= SUM_OPTIMAL_MAX:
            score += 50
            mid = (SUM_OPTIMAL_MIN + SUM_OPTIMAL_MAX) / 2
            score += max(0, 20 - int(abs(total_sum - mid) / 2))
        elif SUM_GOOD_MIN <= total_sum <= SUM_GOOD_MAX:
            score += 25
        else:
            score -= 30

        if ord5 in BALL5_STATS:
            score += BALL5_STATS[ord5]
        gap = ord5 - ord4
        if gap in GAP45_STATS:
            score += GAP45_STATS[gap] // 4
        if ord5 in HOT_BITS:
            score += 10
        elif ord5 in COLD_BITS:
            score -= 3

        candidates.append({
            'ord5': ord5,
            'score': score,
            'is_duplicate': is_duplicate,
            'sum': total_sum
        })

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]


def test_round(data, target_round, max_candidates=15):
    """특정 회차 테스트"""
    actual_data = next((d for d in data if d['round'] == target_round), None)
    if not actual_data:
        return None

    actual = actual_data['balls']
    a1, a2, a3, a4, a5, a6 = actual

    # 학습 데이터
    train_data = [d for d in data if d['round'] < target_round]
    winning_sets = [d['balls'] for d in train_data]

    # ord1, ord6 고정
    ord1, ord6 = a1, a6

    # 각 단계별 순위 확인
    ranks = {}

    # ord2 순위
    ord2_cands = get_ord2_candidates(ord1, ord6, max_candidates=100)
    for i, c in enumerate(ord2_cands):
        if c['ord2'] == a2:
            ranks['ord2'] = i + 1
            break
    else:
        ranks['ord2'] = -1

    # ord3 순위
    ord3_cands = get_ord3_candidates(ord1, a2, ord6, max_candidates=100)
    for i, c in enumerate(ord3_cands):
        if c['ord3'] == a3:
            ranks['ord3'] = i + 1
            break
    else:
        ranks['ord3'] = -1

    # ord4 순위
    ord4_cands = get_ord4_candidates(ord1, a2, a3, ord6, max_candidates=100)
    for i, c in enumerate(ord4_cands):
        if c['ord4'] == a4:
            ranks['ord4'] = i + 1
            break
    else:
        ranks['ord4'] = -1

    # ord5 순위
    ord5_cands = get_ord5_candidates(ord1, a2, a3, a4, ord6, winning_sets, max_candidates=100)
    non_dup = [c for c in ord5_cands if not c['is_duplicate']]
    for i, c in enumerate(non_dup):
        if c['ord5'] == a5:
            ranks['ord5'] = i + 1
            break
    else:
        # 중복으로 제외되었는지 확인
        for c in ord5_cands:
            if c['ord5'] == a5 and c['is_duplicate']:
                ranks['ord5'] = -2  # 중복 제외
                break
        else:
            ranks['ord5'] = -1

    # 모두 top N 안에 있는지
    all_in_top = all(0 < ranks[k] <= max_candidates for k in ['ord2', 'ord3', 'ord4', 'ord5'])

    return {
        'round': target_round,
        'actual': actual,
        'ranks': ranks,
        'all_in_top': all_in_top
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=900, help='시작 회차')
    parser.add_argument('--end', type=int, default=1100, help='종료 회차')
    parser.add_argument('--max', type=int, default=15, help='max_candidates')
    args = parser.parse_args()

    print("=" * 80)
    print(f"백테스트 - 실제 ord1, ord6 고정 + max_candidates={args.max}")
    print(f"범위: {args.start}~{args.end}회차")
    print("=" * 80)

    data = load_data()

    # 통계
    total = 0
    six_match_possible = 0
    failure_stage = {'ord2': 0, 'ord3': 0, 'ord4': 0, 'ord5': 0, 'ord5_dup': 0}

    # 상세 결과
    results = []

    for target_round in range(args.start, args.end + 1):
        result = test_round(data, target_round, args.max)
        if result is None:
            continue

        total += 1
        results.append(result)

        if result['all_in_top']:
            six_match_possible += 1
        else:
            # 첫 탈락 지점
            for stage in ['ord2', 'ord3', 'ord4', 'ord5']:
                rank = result['ranks'][stage]
                if rank == -2:
                    failure_stage['ord5_dup'] += 1
                    break
                elif rank <= 0 or rank > args.max:
                    failure_stage[stage] += 1
                    break

    # 결과 요약
    print(f"\n총 {total}회차 분석")
    print("\n" + "=" * 80)
    print("[결과 요약]")
    print("=" * 80)

    pct = six_match_possible / total * 100 if total > 0 else 0
    print(f"\n✓ 6개 적중 가능: {six_match_possible}회 ({pct:.1f}%)")

    print(f"\n<탈락 단계별 통계>")
    for stage, count in failure_stage.items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {stage:10s}: {count:4d}회 ({pct:5.1f}%) {bar}")

    # 6개 적중 가능한 회차 목록
    six_match_rounds = [r for r in results if r['all_in_top']]
    if six_match_rounds:
        print(f"\n<6개 적중 가능 회차 (총 {len(six_match_rounds)}회)>")
        for r in six_match_rounds[:20]:
            print(f"  {r['round']}회차: {r['actual']} - 순위: ord2={r['ranks']['ord2']}, ord3={r['ranks']['ord3']}, ord4={r['ranks']['ord4']}, ord5={r['ranks']['ord5']}")
        if len(six_match_rounds) > 20:
            print(f"  ... 외 {len(six_match_rounds) - 20}회 더")

    # 아슬아슬하게 탈락한 회차 (순위 16~20)
    print(f"\n<아슬아슬하게 탈락한 회차 (순위 {args.max+1}~{args.max+5})>")
    close_misses = []
    for r in results:
        if not r['all_in_top']:
            max_rank = max(r['ranks'].values())
            if args.max < max_rank <= args.max + 5:
                close_misses.append(r)

    for r in close_misses[:10]:
        print(f"  {r['round']}회차: {r['actual']} - 순위: {r['ranks']}")


if __name__ == "__main__":
    main()
