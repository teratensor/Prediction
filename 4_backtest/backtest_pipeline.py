"""
파이프라인 백테스트

목표 회차를 지정하면 그 이전 데이터만 학습하여 예측 조합 생성 후
실제 당첨번호와 비교

실행: python backtest_pipeline.py --start 900 --end 1000
"""

import csv
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
RESULT_PATH = BASE_DIR / "result" / "result.csv"


# ============ 공통 상수 ============
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

def get_range(num):
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4


# ============ 데이터 로드 ============
def load_all_data():
    """전체 당첨번호 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': tuple(sorted([int(row[f'ball{i}']) for i in range(1, 7)])),
            })
    return sorted(results, key=lambda x: x['round'])


# ============ 1. Firstend ============
def generate_firstend(data, target_round):
    """target_round 이전 데이터로 (ord1, ord6) 쌍 생성"""
    train_data = [d for d in data if d['round'] < target_round]

    # 실제 나온 쌍 빈도
    pair_freq = Counter()
    for r in train_data:
        ord1 = r['balls'][0]
        ord6 = r['balls'][5]
        pair_freq[(ord1, ord6)] += 1

    # 범위 결정
    actual_ord1_max = max(r['balls'][0] for r in train_data)

    # 모든 가능한 쌍
    all_pairs = []
    for o1 in range(1, actual_ord1_max + 1):
        for o6 in range(o1 + 5, 46):
            all_pairs.append((o1, o6))

    seen_pairs = set(pair_freq.keys())
    sorted_pairs = sorted(pair_freq.items(), key=lambda x: -x[1])
    top_freq_pairs = set(p for p, _ in sorted_pairs[:30])
    promising_pairs = set((o1, o6) for (o1, o6) in (set(all_pairs) - seen_pairs) if 1 <= o1 <= 10 and 38 <= o6 <= 45)

    rows = []
    for (o1, o6) in sorted(all_pairs):
        freq = pair_freq.get((o1, o6), 0)
        if (o1, o6) in top_freq_pairs:
            cls = "top_freq"
        elif (o1, o6) in promising_pairs:
            cls = "promising"
        elif (o1, o6) in seen_pairs:
            cls = "seen"
        else:
            cls = "unseen"

        r1, r6 = get_range(o1), get_range(o6)
        rows.append({
            'ord1': o1, 'ord2': '', 'ord3': '', 'ord4': '', 'ord5': '', 'ord6': o6,
            '분류': cls, '빈도': freq,
            'range_code': f"{r1}----{r6}", 'unique_ranges': len({r1, r6}),
            'consecutive': '', 'hot_count': '', 'cold_count': '', 'score': ''
        })
    return rows


# ============ 2. Second ============
# ball2 통계 (내장)
BALL2_STATS = {7: 25, 16: 23, 11: 23, 6: 23, 9: 23, 12: 21, 14: 20, 18: 20, 13: 20, 10: 17, 4: 17, 8: 16, 15: 16, 5: 13, 21: 12, 17: 12, 19: 11, 3: 9, 26: 8, 20: 8, 23: 8, 24: 6, 25: 5, 2: 5}
BALL1_BALL2_STATS = {1: {'avg': 7.6, 'mode': 9}, 2: {'avg': 9.1, 'mode': 6}, 3: {'avg': 9.2, 'mode': 6}, 4: {'avg': 10.4, 'mode': 7}, 5: {'avg': 11.3, 'mode': 12}, 6: {'avg': 12.4, 'mode': 7}, 7: {'avg': 12.4, 'mode': 9}, 8: {'avg': 14.3, 'mode': 11}, 9: {'avg': 14.4, 'mode': 14}, 10: {'avg': 16.5, 'mode': 16}}
GAP12_STATS = {1: 47, 2: 38, 3: 40, 4: 50, 5: 32, 6: 33, 7: 18, 8: 26, 9: 13, 10: 15}
ORD2_BIT_FREQ = {9: 28, 10: 26, 5: 23, 17: 22, 6: 21, 7: 20, 11: 20, 12: 20, 15: 20, 19: 19, 13: 18, 14: 17, 18: 15, 20: 14}

def select_ord2(ord1, ord6, max_candidates=2):
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

def process_second(rows):
    new_rows = []
    for row in rows:
        ord1, ord6 = int(row['ord1']), int(row['ord6'])
        candidates = select_ord2(ord1, ord6)
        if not candidates:
            new_rows.append(row)
        else:
            for cand in candidates:
                new_row = row.copy()
                ord2 = cand['ord2']
                new_row['ord2'] = ord2
                r1, r2, r6 = get_range(ord1), get_range(ord2), get_range(ord6)
                new_row['range_code'] = f"{r1}{r2}---{r6}"
                new_row['unique_ranges'] = len({r1, r2, r6})
                new_row['consecutive'] = 1 if ord2 - ord1 == 1 else 0
                new_row['hot_count'] = sum(1 for x in [ord1, ord2, ord6] if x in HOT_BITS)
                new_row['cold_count'] = sum(1 for x in [ord1, ord2, ord6] if x in COLD_BITS)
                new_rows.append(new_row)
    return new_rows


# ============ 3. Third ============
BALL3_STATS = {17: 25, 18: 23, 19: 22, 12: 21, 14: 21, 20: 20, 16: 20, 13: 19, 21: 18, 22: 18, 15: 17, 11: 16, 23: 15, 24: 14, 10: 13, 25: 12, 9: 11, 26: 10}
GAP23_STATS = {1: 55, 2: 48, 3: 43, 4: 37, 5: 32, 6: 28, 7: 22, 8: 19, 9: 15, 10: 13}
ORD3_BIT_FREQ = {17: 23, 18: 22, 19: 21, 12: 20, 14: 20, 20: 19, 16: 19, 13: 18, 21: 17, 22: 17, 15: 16, 11: 15}
ORD3_RANGE_FREQ = {1: 45.4, 2: 36.7, 0: 5.0, 3: 12.7, 4: 0.3}

def select_ord3(ord1, ord2, ord6, max_candidates=2):
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
        # Range
        ord3_range = get_range(ord3)
        range_freq = ORD3_RANGE_FREQ.get(ord3_range, 0)
        if range_freq >= 30:
            score += int(range_freq * 0.3)
        existing = {get_range(ord1), get_range(ord2), get_range(ord6)}
        if ord3_range not in existing and len(existing) + 1 == 4:
            score += 10
        candidates.append({'ord3': ord3, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]

def process_third(rows):
    new_rows = []
    for row in rows:
        ord1, ord6 = int(row['ord1']), int(row['ord6'])
        ord2 = int(row['ord2']) if row['ord2'] else None
        if ord2 is None:
            new_rows.append(row)
            continue
        candidates = select_ord3(ord1, ord2, ord6)
        if not candidates:
            new_rows.append(row)
        else:
            for cand in candidates:
                new_row = row.copy()
                ord3 = cand['ord3']
                new_row['ord3'] = ord3
                r1, r2, r3, r6 = get_range(ord1), get_range(ord2), get_range(ord3), get_range(ord6)
                new_row['range_code'] = f"{r1}{r2}{r3}--{r6}"
                new_row['unique_ranges'] = len({r1, r2, r3, r6})
                consec = (1 if ord2 - ord1 == 1 else 0) + (1 if ord3 - ord2 == 1 else 0)
                new_row['consecutive'] = consec
                new_row['hot_count'] = sum(1 for x in [ord1, ord2, ord3, ord6] if x in HOT_BITS)
                new_row['cold_count'] = sum(1 for x in [ord1, ord2, ord3, ord6] if x in COLD_BITS)
                new_rows.append(new_row)
    return new_rows


# ============ 4. Fourth ============
BALL4_STATS = {30: 26, 26: 19, 29: 18, 35: 18, 27: 18, 23: 18, 33: 17, 28: 16, 22: 16, 32: 16, 31: 16, 21: 15, 24: 15, 25: 14, 19: 14, 20: 13}
GAP34_STATS = {1: 46, 2: 54, 3: 44, 4: 37, 5: 24, 6: 30, 7: 25, 8: 22, 9: 18, 10: 18}
ORD4_BIT_FREQ = {27: 23, 28: 22, 25: 20, 24: 19, 29: 19, 31: 18, 21: 17, 17: 17, 32: 17, 33: 17, 23: 16}
ORD4_RANGE_FREQ = {2: 42.7, 3: 34.8, 1: 19.3, 4: 2.9, 0: 0.3}
UNEXPECTED_NUMBERS = {14, 37, 15, 36, 13, 34, 17, 40, 39, 41, 11, 12, 43, 42, 8}

def select_ord4(ord1, ord2, ord3, ord6, max_candidates=2):
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
        # Range
        ord4_range = get_range(ord4)
        existing = {get_range(ord1), get_range(ord2), get_range(ord3), get_range(ord6)}
        if ord4_range not in existing and len(existing) + 1 == 4:
            score += 8
        candidates.append({'ord4': ord4, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]

def process_fourth(rows):
    new_rows = []
    for row in rows:
        ord1, ord6 = int(row['ord1']), int(row['ord6'])
        ord2 = int(row['ord2']) if row['ord2'] else None
        ord3 = int(row['ord3']) if row['ord3'] else None
        if ord3 is None:
            new_rows.append(row)
            continue
        candidates = select_ord4(ord1, ord2, ord3, ord6)
        if not candidates:
            new_rows.append(row)
        else:
            for cand in candidates:
                new_row = row.copy()
                ord4 = cand['ord4']
                new_row['ord4'] = ord4
                r1, r2, r3, r4, r6 = get_range(ord1), get_range(ord2), get_range(ord3), get_range(ord4), get_range(ord6)
                new_row['range_code'] = f"{r1}{r2}{r3}{r4}-{r6}"
                new_row['unique_ranges'] = len({r1, r2, r3, r4, r6})
                consec = sum([ord2-ord1==1, ord3-ord2==1, ord4-ord3==1]) if ord2 else 0
                new_row['consecutive'] = consec
                new_row['hot_count'] = sum(1 for x in [ord1, ord2, ord3, ord4, ord6] if x and x in HOT_BITS)
                new_row['cold_count'] = sum(1 for x in [ord1, ord2, ord3, ord4, ord6] if x and x in COLD_BITS)
                new_rows.append(new_row)
    return new_rows


# ============ 5. Fifth ============
BALL5_STATS = {33: 26, 34: 24, 38: 23, 37: 22, 39: 21, 31: 20, 36: 20, 35: 19, 32: 18, 28: 18, 40: 17, 29: 17, 30: 16, 27: 15, 42: 14, 26: 13, 41: 13}
GAP45_STATS = {1: 56, 2: 51, 3: 42, 4: 35, 5: 33, 6: 29, 7: 26, 8: 19, 9: 18, 10: 14}
ORD5_BIT_FREQ = {33: 26, 34: 24, 38: 23, 37: 22, 39: 21, 31: 20, 36: 20, 35: 19, 32: 18, 28: 18}
ORD5_RANGE_FREQ = {3: 54.6, 2: 25.9, 4: 16.1, 1: 3.4, 0: 0.0}
SUM_OPTIMAL_MIN, SUM_OPTIMAL_MAX = 121, 160
SUM_GOOD_MIN, SUM_GOOD_MAX = 100, 170

def check_duplicate(combo, winning_sets, min_match=5):
    combo_set = set(combo)
    for winning in winning_sets:
        if len(combo_set & set(winning)) >= min_match:
            return True
    return False

def select_ord5(ord1, ord2, ord3, ord4, ord6, winning_sets, max_candidates=2):
    min_ord5 = ord4 + 1
    max_ord5 = ord6 - 1
    if min_ord5 > max_ord5:
        return []

    candidates = []
    current_sum = ord1 + ord2 + ord3 + ord4 + ord6

    for ord5 in range(min_ord5, max_ord5 + 1):
        combo = tuple(sorted([ord1, ord2, ord3, ord4, ord5, ord6]))
        # 5개/6개 일치 제외
        if check_duplicate(combo, winning_sets, 5):
            continue

        score = 0
        total_sum = current_sum + ord5

        # Sum (가장 중요)
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
        if ord5 in ORD5_BIT_FREQ:
            score += ORD5_BIT_FREQ[ord5] // 2
        if ord5 in HOT_BITS:
            score += 10
        elif ord5 in COLD_BITS:
            score -= 3
        # Range
        ranges = {get_range(ord1), get_range(ord2), get_range(ord3), get_range(ord4), get_range(ord5), get_range(ord6)}
        if len(ranges) == 4:
            score += 10

        candidates.append({'ord5': ord5, 'score': score, 'sum': total_sum})

    candidates.sort(key=lambda x: -x['score'])
    return candidates[:max_candidates]

def process_fifth(rows, winning_sets):
    new_rows = []
    for row in rows:
        ord1, ord6 = int(row['ord1']), int(row['ord6'])
        ord2 = int(row['ord2']) if row['ord2'] else None
        ord3 = int(row['ord3']) if row['ord3'] else None
        ord4 = int(row['ord4']) if row['ord4'] else None
        if ord4 is None:
            continue
        candidates = select_ord5(ord1, ord2, ord3, ord4, ord6, winning_sets)
        if not candidates:
            continue
        for cand in candidates:
            new_row = row.copy()
            ord5 = cand['ord5']
            new_row['ord5'] = ord5
            r1, r2, r3, r4, r5, r6 = [get_range(x) for x in [ord1, ord2, ord3, ord4, ord5, ord6]]
            new_row['range_code'] = f"{r1}{r2}{r3}{r4}{r5}{r6}"
            new_row['unique_ranges'] = len({r1, r2, r3, r4, r5, r6})
            consec = sum([ord2-ord1==1, ord3-ord2==1, ord4-ord3==1, ord5-ord4==1, ord6-ord5==1])
            new_row['consecutive'] = consec
            new_row['hot_count'] = sum(1 for x in [ord1, ord2, ord3, ord4, ord5, ord6] if x in HOT_BITS)
            new_row['cold_count'] = sum(1 for x in [ord1, ord2, ord3, ord4, ord5, ord6] if x in COLD_BITS)
            new_row['score'] = cand['score']
            new_rows.append(new_row)
    return new_rows


# ============ 메인 백테스트 ============
def run_pipeline(data, target_round):
    """전체 파이프라인 실행"""
    # 학습 데이터 (target_round 이전)
    train_data = [d for d in data if d['round'] < target_round]
    winning_sets = [d['balls'] for d in train_data]

    # 1. Firstend
    rows = generate_firstend(data, target_round)

    # 2. Second
    rows = process_second(rows)

    # 3. Third
    rows = process_third(rows)

    # 4. Fourth
    rows = process_fourth(rows)

    # 5. Fifth
    rows = process_fifth(rows, winning_sets)

    return rows


def evaluate(predictions, actual):
    """예측 조합과 실제 당첨번호 비교"""
    actual_set = set(actual)

    results = []
    for row in predictions:
        combo = tuple(sorted([
            int(row['ord1']), int(row['ord2']), int(row['ord3']),
            int(row['ord4']), int(row['ord5']), int(row['ord6'])
        ]))
        match_count = len(set(combo) & actual_set)
        results.append({
            'combo': combo,
            'match': match_count,
            'score': int(row['score']) if row['score'] else 0,
            '분류': row['분류']
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=900, help='시작 회차')
    parser.add_argument('--end', type=int, default=950, help='종료 회차')
    args = parser.parse_args()

    print(f"백테스트: {args.start}회차 ~ {args.end}회차")
    print(f"전체 조합에서 평가")
    print("=" * 70)

    data = load_all_data()

    # 통계
    total_rounds = 0
    match_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    best_matches = []

    for target_round in range(args.start, args.end + 1):
        # 해당 회차 데이터 확인
        actual_data = next((d for d in data if d['round'] == target_round), None)
        if not actual_data:
            continue

        actual = actual_data['balls']

        # 파이프라인 실행
        predictions = run_pipeline(data, target_round)

        if not predictions:
            print(f"[{target_round}회차] 예측 조합 없음")
            continue

        # 전체 조합 평가
        results = evaluate(predictions, actual)

        # 최고 적중
        best = max(results, key=lambda x: x['match'])
        match_stats[best['match']] += 1
        total_rounds += 1

        best_matches.append({
            'round': target_round,
            'actual': actual,
            'best_combo': best['combo'],
            'match': best['match'],
            'total_combos': len(predictions)
        })

        if best['match'] >= 4:
            print(f"[{target_round}회차] 실제: {actual} | 최고적중: {best['combo']} ({best['match']}개) / {len(predictions)}개 중")

    # 결과 요약
    print("\n" + "=" * 70)
    print(f"[백테스트 결과 요약] ({total_rounds}회차)")
    print("=" * 70)

    for i in range(6, -1, -1):
        cnt = match_stats[i]
        pct = cnt / total_rounds * 100 if total_rounds > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {i}개 적중: {cnt:4d}회 ({pct:5.1f}%) {bar}")

    # 3개 이상 적중률
    hit_3plus = sum(match_stats[i] for i in range(3, 7))
    hit_4plus = sum(match_stats[i] for i in range(4, 7))
    hit_5plus = sum(match_stats[i] for i in range(5, 7))

    print(f"\n  3개+ 적중: {hit_3plus}회 ({hit_3plus/total_rounds*100:.1f}%)")
    print(f"  4개+ 적중: {hit_4plus}회 ({hit_4plus/total_rounds*100:.1f}%)")
    print(f"  5개+ 적중: {hit_5plus}회 ({hit_5plus/total_rounds*100:.1f}%)")


if __name__ == "__main__":
    main()
