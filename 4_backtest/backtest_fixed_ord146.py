"""
백테스트 - 실제 ord1, ord4, ord6 고정 + max_candidates=15

ord1, ord4, ord6를 알고 있다고 가정하고
나머지 ord2, ord3, ord5를 예측하여 6개 적중 가능 여부 확인
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


# ========== 후보 선택 함수들 ==========
BALL2_STATS = {7: 25, 16: 23, 11: 23, 6: 23, 9: 23, 12: 21, 14: 20, 18: 20, 13: 20, 10: 17, 4: 17, 8: 16, 15: 16, 5: 13, 21: 12, 17: 12, 19: 11, 3: 9, 26: 8, 20: 8, 23: 8, 24: 6, 25: 5, 2: 5}
BALL1_BALL2_STATS = {1: {'avg': 7.6, 'mode': 9}, 2: {'avg': 9.1, 'mode': 6}, 3: {'avg': 9.2, 'mode': 6}, 4: {'avg': 10.4, 'mode': 7}, 5: {'avg': 11.3, 'mode': 12}, 6: {'avg': 12.4, 'mode': 7}, 7: {'avg': 12.4, 'mode': 9}, 8: {'avg': 14.3, 'mode': 11}, 9: {'avg': 14.4, 'mode': 14}, 10: {'avg': 16.5, 'mode': 16}}
GAP12_STATS = {1: 47, 2: 38, 3: 40, 4: 50, 5: 32, 6: 33, 7: 18, 8: 26, 9: 13, 10: 15}
ORD2_BIT_FREQ = {9: 28, 10: 26, 5: 23, 17: 22, 6: 21, 7: 20, 11: 20, 12: 20, 15: 20, 19: 19, 13: 18, 14: 17, 18: 15, 20: 14}

def get_ord2_rank(ord1, ord2, ord4):
    """ord2의 순위 반환 (ord4를 상한으로 사용)"""
    min_ord2 = ord1 + 1
    max_ord2 = ord4 - 2  # ord3 자리 확보
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

def get_ord3_rank(ord2, ord3, ord4):
    """ord3의 순위 반환"""
    min_ord3 = ord2 + 1
    max_ord3 = ord4 - 1
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


BALL5_STATS = {33: 26, 34: 24, 38: 23, 37: 22, 39: 21, 31: 20, 36: 20, 35: 19, 32: 18, 28: 18, 40: 17, 29: 17, 30: 16, 27: 15, 42: 14, 26: 13, 41: 13}
GAP45_STATS = {1: 56, 2: 51, 3: 42, 4: 35, 5: 33, 6: 29, 7: 26, 8: 19, 9: 18, 10: 14}
SUM_OPTIMAL_MIN, SUM_OPTIMAL_MAX = 121, 160
SUM_GOOD_MIN, SUM_GOOD_MAX = 100, 170

def get_ord5_rank(ord1, ord2, ord3, ord4, ord5, ord6):
    """ord5의 순위 반환"""
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
    """각 단계별 실제 번호의 순위 반환 (ord1, ord4, ord6 고정)"""
    a1, a2, a3, a4, a5, a6 = actual

    return {
        'ord2': get_ord2_rank(a1, a2, a4),
        'ord3': get_ord3_rank(a2, a3, a4),
        'ord5': get_ord5_rank(a1, a2, a3, a4, a5, a6),
    }


def main():
    print("=" * 90)
    print("백테스트 - ord1, ord4, ord6 고정 + max_candidates=15")
    print("조합 수: 15 * 15 * 15 = 3,375개 (단일 firstend 쌍당)")
    print("=" * 90)

    data = load_data()

    # 모든 회차의 순위 수집
    all_ranks = []
    for d in data:
        if 900 <= d['round'] <= 1200:
            ranks = analyze_round(d['balls'])
            if all(r > 0 for r in ranks.values()):
                all_ranks.append({
                    'round': d['round'],
                    'actual': d['balls'],
                    'ranks': ranks,
                    'max_rank': max(ranks.values())
                })
            else:
                # 일부 순위가 유효하지 않은 경우
                all_ranks.append({
                    'round': d['round'],
                    'actual': d['balls'],
                    'ranks': ranks,
                    'max_rank': 999  # 탈락 처리
                })

    print(f"\n분석 대상: {len(all_ranks)}회차 (900~1200)")

    # 다양한 max_candidates 값 테스트
    test_values = [2, 3, 5, 7, 10, 12, 15, 20]

    print("\n" + "=" * 90)
    print(f"{'max_cand':>8} | {'6개적중':>8} | {'적중률':>8} | {'조합수':>12} | 비고")
    print("-" * 90)

    for mc in test_values:
        hit_count = sum(1 for r in all_ranks if r['max_rank'] <= mc)
        hit_rate = hit_count / len(all_ranks) * 100
        combos = mc ** 3  # ord2 * ord3 * ord5

        note = ""
        if hit_rate >= 80:
            note = "★★★ 매우 높음"
        elif hit_rate >= 70:
            note = "★★ 권장"
        elif hit_rate >= 60:
            note = "★ 균형"

        print(f"{mc:>8} | {hit_count:>6}회 | {hit_rate:>6.1f}% | {combos:>12,} | {note}")

    # max_candidates=15 상세 분석
    mc = 15
    hit_count = sum(1 for r in all_ranks if r['max_rank'] <= mc)
    hit_rate = hit_count / len(all_ranks) * 100

    print("\n" + "=" * 90)
    print(f"[max_candidates=15 상세 분석]")
    print("=" * 90)
    print(f"\n6개 적중 가능: {hit_count}회 ({hit_rate:.1f}%)")
    print(f"조합 수: 15^3 = 3,375개")

    # 탈락 단계별 통계
    failure_stats = {'ord2': 0, 'ord3': 0, 'ord5': 0}
    for r in all_ranks:
        if r['max_rank'] > mc:
            for stage in ['ord2', 'ord3', 'ord5']:
                if r['ranks'][stage] > mc or r['ranks'][stage] <= 0:
                    failure_stats[stage] += 1
                    break

    print(f"\n<탈락 단계별 통계>")
    for stage, count in failure_stats.items():
        pct = count / len(all_ranks) * 100
        print(f"  {stage}: {count}회 ({pct:.1f}%)")

    # 단계별 순위 분포
    print(f"\n<단계별 순위 분포>")
    for stage in ['ord2', 'ord3', 'ord5']:
        valid_ranks = [r['ranks'][stage] for r in all_ranks if r['ranks'][stage] > 0]
        if valid_ranks:
            avg = sum(valid_ranks) / len(valid_ranks)
            sorted_ranks = sorted(valid_ranks)
            p50 = sorted_ranks[len(sorted_ranks) // 2]
            p90 = sorted_ranks[int(len(sorted_ranks) * 0.90)]
            max_r = max(valid_ranks)

            top5 = sum(1 for r in valid_ranks if r <= 5) / len(valid_ranks) * 100
            top10 = sum(1 for r in valid_ranks if r <= 10) / len(valid_ranks) * 100
            top15 = sum(1 for r in valid_ranks if r <= 15) / len(valid_ranks) * 100

            print(f"\n  {stage}:")
            print(f"    평균: {avg:.1f}, 중앙값: {p50}, 90%: {p90}, 최대: {max_r}")
            print(f"    top5: {top5:.1f}%, top10: {top10:.1f}%, top15: {top15:.1f}%")

    # 6개 적중 가능 회차 샘플
    hit_rounds = [r for r in all_ranks if r['max_rank'] <= 15]
    print(f"\n<6개 적중 가능 회차 샘플 (총 {len(hit_rounds)}회)>")
    for r in hit_rounds[:10]:
        print(f"  {r['round']}회차: {r['actual']} - 순위: {r['ranks']}")


if __name__ == "__main__":
    main()
