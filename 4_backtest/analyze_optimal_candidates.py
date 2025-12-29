"""
최적의 max_candidates 값 분석

다양한 max_candidates 값에 대해:
1. 6개 적중 가능 비율
2. 예상 조합 수
3. 효율성 (적중률 / 조합수)
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"

# 상수
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

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


# ========== 후보 선택 함수들 (점수만 계산) ==========
BALL2_STATS = {7: 25, 16: 23, 11: 23, 6: 23, 9: 23, 12: 21, 14: 20, 18: 20, 13: 20, 10: 17, 4: 17, 8: 16, 15: 16, 5: 13, 21: 12, 17: 12, 19: 11, 3: 9, 26: 8, 20: 8, 23: 8, 24: 6, 25: 5, 2: 5}
BALL1_BALL2_STATS = {1: {'avg': 7.6, 'mode': 9}, 2: {'avg': 9.1, 'mode': 6}, 3: {'avg': 9.2, 'mode': 6}, 4: {'avg': 10.4, 'mode': 7}, 5: {'avg': 11.3, 'mode': 12}, 6: {'avg': 12.4, 'mode': 7}, 7: {'avg': 12.4, 'mode': 9}, 8: {'avg': 14.3, 'mode': 11}, 9: {'avg': 14.4, 'mode': 14}, 10: {'avg': 16.5, 'mode': 16}}
GAP12_STATS = {1: 47, 2: 38, 3: 40, 4: 50, 5: 32, 6: 33, 7: 18, 8: 26, 9: 13, 10: 15}
ORD2_BIT_FREQ = {9: 28, 10: 26, 5: 23, 17: 22, 6: 21, 7: 20, 11: 20, 12: 20, 15: 20, 19: 19, 13: 18, 14: 17, 18: 15, 20: 14}

def get_ord2_rank(ord1, ord2, ord6):
    min_ord2 = ord1 + 1
    max_ord2 = ord6 - 4
    if min_ord2 > max_ord2 or ord2 < min_ord2 or ord2 > max_ord2:
        return -1

    candidates = []
    for o2 in range(min_ord2, max_ord2 + 1):
        score = 0
        if o2 in BALL2_STATS:
            score += BALL2_STATS[o2] * 2
        if ord1 in BALL1_BALL2_STATS:
            stats = BALL1_BALL2_STATS[ord1]
            if o2 == stats['mode']:
                score += 15
            if abs(o2 - stats['avg']) <= 2:
                score += 10
        gap = o2 - ord1
        if gap in GAP12_STATS:
            score += GAP12_STATS[gap] // 2
        if 10 <= o2 <= 19:
            score += 10
        if o2 in ORD2_BIT_FREQ:
            score += ORD2_BIT_FREQ[o2]
        if o2 in HOT_BITS:
            score += 10
        elif o2 in COLD_BITS:
            score -= 5
        candidates.append((o2, score))

    candidates.sort(key=lambda x: -x[1])
    for i, (o, _) in enumerate(candidates):
        if o == ord2:
            return i + 1
    return -1


BALL3_STATS = {17: 25, 18: 23, 19: 22, 12: 21, 14: 21, 20: 20, 16: 20, 13: 19, 21: 18, 22: 18, 15: 17, 11: 16, 23: 15, 24: 14, 10: 13, 25: 12, 9: 11, 26: 10}
GAP23_STATS = {1: 55, 2: 48, 3: 43, 4: 37, 5: 32, 6: 28, 7: 22, 8: 19, 9: 15, 10: 13}
ORD3_BIT_FREQ = {17: 23, 18: 22, 19: 21, 12: 20, 14: 20, 20: 19, 16: 19, 13: 18, 21: 17, 22: 17, 15: 16, 11: 15}

def get_ord3_rank(ord1, ord2, ord3, ord6):
    min_ord3 = ord2 + 1
    max_ord3 = ord6 - 3
    if min_ord3 > max_ord3 or ord3 < min_ord3 or ord3 > max_ord3:
        return -1

    candidates = []
    for o3 in range(min_ord3, max_ord3 + 1):
        score = 0
        if o3 in BALL3_STATS:
            score += BALL3_STATS[o3] * 2
        gap = o3 - ord2
        if gap in GAP23_STATS:
            score += GAP23_STATS[gap] // 2
        if 10 <= o3 <= 19:
            score += 10
        if o3 in ORD3_BIT_FREQ:
            score += ORD3_BIT_FREQ[o3]
        if o3 in HOT_BITS:
            score += 10
        elif o3 in COLD_BITS:
            score -= 5
        candidates.append((o3, score))

    candidates.sort(key=lambda x: -x[1])
    for i, (o, _) in enumerate(candidates):
        if o == ord3:
            return i + 1
    return -1


BALL4_STATS = {30: 26, 26: 19, 29: 18, 35: 18, 27: 18, 23: 18, 33: 17, 28: 16, 22: 16, 32: 16, 31: 16, 21: 15, 24: 15, 25: 14, 19: 14, 20: 13}
GAP34_STATS = {1: 46, 2: 54, 3: 44, 4: 37, 5: 24, 6: 30, 7: 25, 8: 22, 9: 18, 10: 18}
ORD4_BIT_FREQ = {27: 23, 28: 22, 25: 20, 24: 19, 29: 19, 31: 18, 21: 17, 17: 17, 32: 17, 33: 17, 23: 16}
UNEXPECTED_NUMBERS = {14, 37, 15, 36, 13, 34, 17, 40, 39, 41, 11, 12, 43, 42, 8}

def get_ord4_rank(ord1, ord2, ord3, ord4, ord6):
    min_ord4 = ord3 + 1
    max_ord4 = ord6 - 1
    if min_ord4 > max_ord4 or ord4 < min_ord4 or ord4 > max_ord4:
        return -1

    candidates = []
    for o4 in range(min_ord4, max_ord4 + 1):
        score = 0
        if o4 in BALL4_STATS:
            score += BALL4_STATS[o4]
        if o4 in UNEXPECTED_NUMBERS:
            score += 15
        if 20 <= o4 <= 29:
            score += 8
        elif 30 <= o4 <= 39:
            score += 6
        gap = o4 - ord3
        if gap in GAP34_STATS:
            score += GAP34_STATS[gap] // 4
        if o4 in ORD4_BIT_FREQ:
            score += ORD4_BIT_FREQ[o4] // 2
        if o4 in COLD_BITS:
            score += 12
        elif o4 in HOT_BITS:
            score -= 3
        candidates.append((o4, score))

    candidates.sort(key=lambda x: -x[1])
    for i, (o, _) in enumerate(candidates):
        if o == ord4:
            return i + 1
    return -1


BALL5_STATS = {33: 26, 34: 24, 38: 23, 37: 22, 39: 21, 31: 20, 36: 20, 35: 19, 32: 18, 28: 18, 40: 17, 29: 17, 30: 16, 27: 15, 42: 14, 26: 13, 41: 13}
GAP45_STATS = {1: 56, 2: 51, 3: 42, 4: 35, 5: 33, 6: 29, 7: 26, 8: 19, 9: 18, 10: 14}
SUM_OPTIMAL_MIN, SUM_OPTIMAL_MAX = 121, 160
SUM_GOOD_MIN, SUM_GOOD_MAX = 100, 170

def get_ord5_rank(ord1, ord2, ord3, ord4, ord5, ord6):
    min_ord5 = ord4 + 1
    max_ord5 = ord6 - 1
    if min_ord5 > max_ord5 or ord5 < min_ord5 or ord5 > max_ord5:
        return -1

    current_sum = ord1 + ord2 + ord3 + ord4 + ord6
    candidates = []

    for o5 in range(min_ord5, max_ord5 + 1):
        score = 0
        total_sum = current_sum + o5

        if SUM_OPTIMAL_MIN <= total_sum <= SUM_OPTIMAL_MAX:
            score += 50
            mid = (SUM_OPTIMAL_MIN + SUM_OPTIMAL_MAX) / 2
            score += max(0, 20 - int(abs(total_sum - mid) / 2))
        elif SUM_GOOD_MIN <= total_sum <= SUM_GOOD_MAX:
            score += 25
        else:
            score -= 30

        if o5 in BALL5_STATS:
            score += BALL5_STATS[o5]
        gap = o5 - ord4
        if gap in GAP45_STATS:
            score += GAP45_STATS[gap] // 4
        if o5 in HOT_BITS:
            score += 10
        elif o5 in COLD_BITS:
            score -= 3

        candidates.append((o5, score))

    candidates.sort(key=lambda x: -x[1])
    for i, (o, _) in enumerate(candidates):
        if o == ord5:
            return i + 1
    return -1


def analyze_round(actual):
    """각 단계별 실제 번호의 순위 반환"""
    a1, a2, a3, a4, a5, a6 = actual

    return {
        'ord2': get_ord2_rank(a1, a2, a6),
        'ord3': get_ord3_rank(a1, a2, a3, a6),
        'ord4': get_ord4_rank(a1, a2, a3, a4, a6),
        'ord5': get_ord5_rank(a1, a2, a3, a4, a5, a6),
    }


def main():
    print("=" * 90)
    print("최적 max_candidates 분석 (900~1200회차, 301회)")
    print("=" * 90)

    data = load_data()

    # 모든 회차의 순위 수집
    all_ranks = []
    for d in data:
        if 900 <= d['round'] <= 1200:
            ranks = analyze_round(d['balls'])
            if all(r > 0 for r in ranks.values()):  # 모든 순위가 유효한 경우만
                all_ranks.append({
                    'round': d['round'],
                    'actual': d['balls'],
                    'ranks': ranks,
                    'max_rank': max(ranks.values())
                })

    print(f"\n분석 대상: {len(all_ranks)}회차")

    # 다양한 max_candidates 값 테스트
    test_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]

    print("\n" + "=" * 90)
    print(f"{'max_cand':>8} | {'6개적중':>8} | {'적중률':>8} | {'조합수(추정)':>12} | {'효율성':>10} | 비고")
    print("-" * 90)

    results = []
    for mc in test_values:
        # 6개 적중 가능 회차 수
        hit_count = sum(1 for r in all_ranks if r['max_rank'] <= mc)
        hit_rate = hit_count / len(all_ranks) * 100

        # 조합 수 추정 (firstend 쌍 약 300개 가정)
        # 각 단계 평균 후보 수를 고려하여 추정
        # 실제로는 범위에 따라 다르지만 대략적으로 계산
        estimated_combos = 300 * (mc ** 4)  # firstend * ord2 * ord3 * ord4 * ord5

        # 효율성: 적중률 / log(조합수)
        import math
        efficiency = hit_rate / math.log10(estimated_combos) if estimated_combos > 0 else 0

        # 비고
        note = ""
        if mc == 2:
            note = "현재 설정"
        elif hit_rate >= 50 and hit_rate < 60:
            note = "절반 이상"
        elif hit_rate >= 60 and hit_rate < 70:
            note = "★ 균형점"
        elif hit_rate >= 70 and hit_rate < 80:
            note = "★★ 권장"
        elif hit_rate >= 80:
            note = "높은 적중률"

        results.append({
            'max_cand': mc,
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'combos': estimated_combos,
            'efficiency': efficiency
        })

        print(f"{mc:>8} | {hit_count:>6}회 | {hit_rate:>6.1f}% | {estimated_combos:>12,} | {efficiency:>10.2f} | {note}")

    # 최적 값 분석
    print("\n" + "=" * 90)
    print("[분석 결과]")
    print("=" * 90)

    # 효율성 기준 최적
    best_eff = max(results, key=lambda x: x['efficiency'])
    print(f"\n1. 효율성 최고: max_candidates={best_eff['max_cand']}")
    print(f"   - 적중률: {best_eff['hit_rate']:.1f}%, 조합수: {best_eff['combos']:,}개")

    # 적중률 50% 이상 중 최소 조합
    over_50 = [r for r in results if r['hit_rate'] >= 50]
    if over_50:
        min_combo_50 = min(over_50, key=lambda x: x['combos'])
        print(f"\n2. 50%+ 적중 중 최소 조합: max_candidates={min_combo_50['max_cand']}")
        print(f"   - 적중률: {min_combo_50['hit_rate']:.1f}%, 조합수: {min_combo_50['combos']:,}개")

    # 적중률 70% 이상 중 최소 조합
    over_70 = [r for r in results if r['hit_rate'] >= 70]
    if over_70:
        min_combo_70 = min(over_70, key=lambda x: x['combos'])
        print(f"\n3. 70%+ 적중 중 최소 조합: max_candidates={min_combo_70['max_cand']}")
        print(f"   - 적중률: {min_combo_70['hit_rate']:.1f}%, 조합수: {min_combo_70['combos']:,}개")

    # 단계별 순위 분포 분석
    print("\n" + "=" * 90)
    print("[단계별 순위 분포]")
    print("=" * 90)

    for stage in ['ord2', 'ord3', 'ord4', 'ord5']:
        ranks = [r['ranks'][stage] for r in all_ranks]
        avg = sum(ranks) / len(ranks)

        # 분위수
        sorted_ranks = sorted(ranks)
        p50 = sorted_ranks[len(sorted_ranks) // 2]
        p75 = sorted_ranks[int(len(sorted_ranks) * 0.75)]
        p90 = sorted_ranks[int(len(sorted_ranks) * 0.90)]
        max_r = max(ranks)

        print(f"\n{stage}:")
        print(f"  평균: {avg:.1f}, 중앙값: {p50}, 75%: {p75}, 90%: {p90}, 최대: {max_r}")

        # 각 구간별 비율
        for threshold in [5, 10, 15, 20]:
            rate = sum(1 for r in ranks if r <= threshold) / len(ranks) * 100
            print(f"  top{threshold:2d} 포함률: {rate:5.1f}%")


if __name__ == "__main__":
    main()
