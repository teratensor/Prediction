"""
900회차 테스트 - 첫수=1, 끝수=42 고정, max_candidates=15

실제 당첨번호: (1, 9, 10, 14, 33, 42)
"""

import csv
from pathlib import Path

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


def main():
    print("=" * 80)
    print("900회차 테스트 - 첫수=1, 끝수=42 고정, max_candidates=15")
    print("실제 당첨번호: (1, 9, 10, 14, 33, 42)")
    print("=" * 80)

    data = load_data()

    # 899회차까지 학습 데이터
    train_data = [d for d in data if d['round'] < 900]
    winning_sets = [d['balls'] for d in train_data]

    # 실제 당첨번호
    actual = (1, 9, 10, 14, 33, 42)
    ord1, ord6 = 1, 42

    print(f"\n학습 데이터: {len(train_data)}회차 (1~899)")

    # ========== 2단계: ord2 후보 ==========
    print("\n" + "=" * 80)
    print("[2단계] ord2 후보 (ord1=1, ord6=42)")
    print("=" * 80)

    ord2_candidates = get_ord2_candidates(ord1, ord6, max_candidates=15)
    print(f"후보 수: {len(ord2_candidates)}개")
    print("\n순위 | ord2 | 점수 | 실제?")
    print("-" * 40)

    actual_ord2_rank = -1
    for i, c in enumerate(ord2_candidates):
        is_actual = "★ 실제" if c['ord2'] == 9 else ""
        print(f"{i+1:4d} | {c['ord2']:4d} | {c['score']:4d} | {is_actual}")
        if c['ord2'] == 9:
            actual_ord2_rank = i + 1

    print(f"\n→ 실제 ord2=9의 순위: {actual_ord2_rank}")

    # ========== 3단계: ord3 후보 (ord2=9 고정) ==========
    print("\n" + "=" * 80)
    print("[3단계] ord3 후보 (ord1=1, ord2=9, ord6=42)")
    print("=" * 80)

    ord3_candidates = get_ord3_candidates(ord1, 9, ord6, max_candidates=15)
    print(f"후보 수: {len(ord3_candidates)}개")
    print("\n순위 | ord3 | 점수 | 실제?")
    print("-" * 40)

    actual_ord3_rank = -1
    for i, c in enumerate(ord3_candidates):
        is_actual = "★ 실제" if c['ord3'] == 10 else ""
        print(f"{i+1:4d} | {c['ord3']:4d} | {c['score']:4d} | {is_actual}")
        if c['ord3'] == 10:
            actual_ord3_rank = i + 1

    print(f"\n→ 실제 ord3=10의 순위: {actual_ord3_rank}")

    # ========== 4단계: ord4 후보 (ord3=10 고정) ==========
    print("\n" + "=" * 80)
    print("[4단계] ord4 후보 (ord1=1, ord2=9, ord3=10, ord6=42)")
    print("=" * 80)

    ord4_candidates = get_ord4_candidates(ord1, 9, 10, ord6, max_candidates=15)
    print(f"후보 수: {len(ord4_candidates)}개")
    print("\n순위 | ord4 | 점수 | 실제?")
    print("-" * 40)

    actual_ord4_rank = -1
    for i, c in enumerate(ord4_candidates):
        is_actual = "★ 실제" if c['ord4'] == 14 else ""
        print(f"{i+1:4d} | {c['ord4']:4d} | {c['score']:4d} | {is_actual}")
        if c['ord4'] == 14:
            actual_ord4_rank = i + 1

    print(f"\n→ 실제 ord4=14의 순위: {actual_ord4_rank}")

    # ========== 5단계: ord5 후보 (ord4=14 고정) ==========
    print("\n" + "=" * 80)
    print("[5단계] ord5 후보 (ord1=1, ord2=9, ord3=10, ord4=14, ord6=42)")
    print("=" * 80)

    ord5_candidates = get_ord5_candidates(ord1, 9, 10, 14, ord6, winning_sets, max_candidates=15)
    print(f"후보 수: {len(ord5_candidates)}개")
    print("\n순위 | ord5 | 점수 | 합계 | 중복? | 실제?")
    print("-" * 60)

    actual_ord5_rank = -1
    for i, c in enumerate(ord5_candidates):
        is_actual = "★ 실제" if c['ord5'] == 33 else ""
        dup = "중복" if c['is_duplicate'] else ""
        print(f"{i+1:4d} | {c['ord5']:4d} | {c['score']:4d} | {c['sum']:4d} | {dup:4s} | {is_actual}")
        if c['ord5'] == 33:
            actual_ord5_rank = i + 1

    print(f"\n→ 실제 ord5=33의 순위: {actual_ord5_rank}")

    # ========== 결과 요약 ==========
    print("\n" + "=" * 80)
    print("[결과 요약]")
    print("=" * 80)
    print(f"실제 당첨번호: {actual}")
    print(f"\n각 단계별 실제 번호 순위:")
    print(f"  ord2=9  → {actual_ord2_rank}위")
    print(f"  ord3=10 → {actual_ord3_rank}위")
    print(f"  ord4=14 → {actual_ord4_rank}위")
    print(f"  ord5=33 → {actual_ord5_rank}위")

    # 6개 적중 가능 여부
    all_in_top15 = all([
        actual_ord2_rank != -1 and actual_ord2_rank <= 15,
        actual_ord3_rank != -1 and actual_ord3_rank <= 15,
        actual_ord4_rank != -1 and actual_ord4_rank <= 15,
        actual_ord5_rank != -1 and actual_ord5_rank <= 15,
    ])

    if all_in_top15:
        print(f"\n✓ 모든 번호가 top15 안에 있음 → 6개 적중 가능!")

        # 실제로 조합 생성하여 확인
        print("\n전체 조합 생성 중...")
        all_combos = []

        for c2 in ord2_candidates:
            ord2 = c2['ord2']
            for c3 in get_ord3_candidates(ord1, ord2, ord6, 15):
                ord3 = c3['ord3']
                for c4 in get_ord4_candidates(ord1, ord2, ord3, ord6, 15):
                    ord4 = c4['ord4']
                    for c5 in get_ord5_candidates(ord1, ord2, ord3, ord4, ord6, winning_sets, 15):
                        if not c5['is_duplicate']:
                            ord5 = c5['ord5']
                            combo = tuple(sorted([ord1, ord2, ord3, ord4, ord5, ord6]))
                            all_combos.append(combo)

        print(f"총 조합 수: {len(all_combos):,}개")

        # 실제 당첨번호 포함 여부
        if actual in all_combos:
            idx = all_combos.index(actual)
            print(f"\n★★★ 실제 당첨번호 (1, 9, 10, 14, 33, 42) 포함됨! (인덱스: {idx})")
        else:
            print(f"\n✗ 실제 당첨번호 미포함")

            # 가장 많이 일치하는 조합 찾기
            best_match = 0
            best_combo = None
            for combo in all_combos:
                match = len(set(actual) & set(combo))
                if match > best_match:
                    best_match = match
                    best_combo = combo
            print(f"  최고 일치: {best_combo} ({best_match}개 일치)")
    else:
        print(f"\n✗ 일부 번호가 top15 밖 → 6개 적중 불가능")


if __name__ == "__main__":
    main()
