"""
ord4 (넷째수) 특이성 및 의존성 분석

- ord1, ord6와의 관계
- 구간별 분포
- 조건부 확률
- 패턴 분석
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "1_data" / "winning_numbers.csv"


def load_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = tuple(sorted([int(row[f'ball{i}']) for i in range(1, 7)]))
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'ord1': balls[0],
                'ord2': balls[1],
                'ord3': balls[2],
                'ord4': balls[3],
                'ord5': balls[4],
                'ord6': balls[5],
            })
    return sorted(results, key=lambda x: x['round'])


def get_range(num):
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4


def analyze(data):
    total = len(data)
    print("=" * 80)
    print("ord4 (넷째수) 특이성 및 의존성 분석")
    print("=" * 80)
    print(f"총 {total}회차 분석\n")

    # ========== 1. 기본 통계 ==========
    ord4_values = [d['ord4'] for d in data]
    ord4_freq = Counter(ord4_values)

    print("[1] 기본 통계")
    print("-" * 40)
    print(f"범위: {min(ord4_values)} ~ {max(ord4_values)}")
    print(f"평균: {sum(ord4_values)/len(ord4_values):.1f}")
    print(f"최빈값: {ord4_freq.most_common(1)[0]}")
    print(f"상위 10개: {ord4_freq.most_common(10)}")

    # ========== 2. 구간별 분포 ==========
    print("\n[2] 구간별 분포")
    print("-" * 40)
    range_freq = Counter(get_range(d['ord4']) for d in data)
    range_names = {0: '01-09', 1: '10-19', 2: '20-29', 3: '30-39', 4: '40-45'}
    for r in range(5):
        freq = range_freq.get(r, 0)
        pct = freq / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {range_names[r]}: {freq:3d}회 ({pct:5.1f}%) {bar}")

    # ========== 3. ord1과의 관계 ==========
    print("\n[3] ord1 → ord4 의존성")
    print("-" * 40)

    # ord1 구간별 ord4 평균
    ord1_to_ord4 = defaultdict(list)
    for d in data:
        ord1_range = get_range(d['ord1'])
        ord1_to_ord4[ord1_range].append(d['ord4'])

    print("ord1 구간별 ord4 평균:")
    for r in range(5):
        if ord1_to_ord4[r]:
            avg = sum(ord1_to_ord4[r]) / len(ord1_to_ord4[r])
            print(f"  ord1={range_names[r]}: ord4 평균 = {avg:.1f}")

    # ord1 값별 ord4 최빈값
    ord1_to_ord4_freq = defaultdict(Counter)
    for d in data:
        ord1_to_ord4_freq[d['ord1']][d['ord4']] += 1

    print("\nord1 값별 ord4 최빈값 (상위 5개 ord1):")
    ord1_freq = Counter(d['ord1'] for d in data)
    for ord1, _ in ord1_freq.most_common(5):
        mode = ord1_to_ord4_freq[ord1].most_common(3)
        print(f"  ord1={ord1}: ord4 최빈 = {mode}")

    # ========== 4. ord6과의 관계 ==========
    print("\n[4] ord6 → ord4 의존성")
    print("-" * 40)

    # ord6 구간별 ord4 평균
    ord6_to_ord4 = defaultdict(list)
    for d in data:
        ord6_range = get_range(d['ord6'])
        ord6_to_ord4[ord6_range].append(d['ord4'])

    print("ord6 구간별 ord4 평균:")
    for r in range(5):
        if ord6_to_ord4[r]:
            avg = sum(ord6_to_ord4[r]) / len(ord6_to_ord4[r])
            print(f"  ord6={range_names[r]}: ord4 평균 = {avg:.1f}")

    # ========== 5. ord1+ord6 조합별 ord4 ==========
    print("\n[5] (ord1, ord6) 조합별 ord4 분포")
    print("-" * 40)

    pair_to_ord4 = defaultdict(list)
    for d in data:
        pair_to_ord4[(d['ord1'], d['ord6'])].append(d['ord4'])

    # 빈도 높은 쌍에서 ord4 분석
    pair16_freq = Counter((d['ord1'], d['ord6']) for d in data)
    print("상위 (ord1, ord6) 쌍별 ord4:")
    for (o1, o6), cnt in pair16_freq.most_common(10):
        ord4_list = pair_to_ord4[(o1, o6)]
        avg = sum(ord4_list) / len(ord4_list)
        mode = Counter(ord4_list).most_common(1)[0] if ord4_list else (0, 0)
        print(f"  ({o1:2d}, {o6:2d}): {cnt}회, ord4 평균={avg:.1f}, 최빈={mode[0]}")

    # ========== 6. 간격 분석 ==========
    print("\n[6] 간격 분석")
    print("-" * 40)

    # ord3-ord4 간격
    gap34 = [d['ord4'] - d['ord3'] for d in data]
    gap34_freq = Counter(gap34)
    print("ord3 → ord4 간격:")
    for gap, freq in gap34_freq.most_common(10):
        pct = freq / total * 100
        print(f"  gap={gap:2d}: {freq:3d}회 ({pct:5.1f}%)")

    # ord4-ord5 간격
    gap45 = [d['ord5'] - d['ord4'] for d in data]
    gap45_freq = Counter(gap45)
    print("\nord4 → ord5 간격:")
    for gap, freq in gap45_freq.most_common(10):
        pct = freq / total * 100
        print(f"  gap={gap:2d}: {freq:3d}회 ({pct:5.1f}%)")

    # ========== 7. ord4 위치 비율 ==========
    print("\n[7] ord4의 상대적 위치")
    print("-" * 40)

    # ord4가 (ord1~ord6) 범위에서 어느 위치인지
    positions = []
    for d in data:
        span = d['ord6'] - d['ord1']
        if span > 0:
            pos = (d['ord4'] - d['ord1']) / span  # 0~1 사이
            positions.append(pos)

    avg_pos = sum(positions) / len(positions)
    print(f"ord4의 평균 상대 위치: {avg_pos:.3f}")
    print(f"  (0=ord1에 가까움, 1=ord6에 가까움)")
    print(f"  이론적 위치: 0.6 (4번째/6개)")

    # 위치 구간별 분포
    pos_ranges = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print("\n상대 위치 분포:")
    for low, high in pos_ranges:
        cnt = sum(1 for p in positions if low <= p < high)
        pct = cnt / len(positions) * 100
        print(f"  {low:.1f}~{high:.1f}: {cnt:3d}회 ({pct:5.1f}%)")

    # ========== 8. 연속수 패턴 ==========
    print("\n[8] ord4 연속수 패턴")
    print("-" * 40)

    # ord4가 연속수에 포함되는 경우
    consec_with_ord3 = sum(1 for d in data if d['ord4'] - d['ord3'] == 1)
    consec_with_ord5 = sum(1 for d in data if d['ord5'] - d['ord4'] == 1)
    both_consec = sum(1 for d in data if d['ord4'] - d['ord3'] == 1 and d['ord5'] - d['ord4'] == 1)

    print(f"ord3-ord4 연속: {consec_with_ord3}회 ({consec_with_ord3/total*100:.1f}%)")
    print(f"ord4-ord5 연속: {consec_with_ord5}회 ({consec_with_ord5/total*100:.1f}%)")
    print(f"ord3-ord4-ord5 연속: {both_consec}회 ({both_consec/total*100:.1f}%)")

    # ========== 9. ord4 예측 공식 탐색 ==========
    print("\n[9] ord4 예측 공식 탐색")
    print("-" * 40)

    # ord4 ≈ a*ord1 + b*ord6 + c 형태 탐색
    # 단순 선형 근사
    sum_x1, sum_x6, sum_y = 0, 0, 0
    sum_x1y, sum_x6y, sum_x1x6 = 0, 0, 0
    sum_x1_sq, sum_x6_sq = 0, 0

    for d in data:
        x1, x6, y = d['ord1'], d['ord6'], d['ord4']
        sum_x1 += x1
        sum_x6 += x6
        sum_y += y
        sum_x1y += x1 * y
        sum_x6y += x6 * y
        sum_x1x6 += x1 * x6
        sum_x1_sq += x1 * x1
        sum_x6_sq += x6 * x6

    # 단순 평균 기반 공식
    avg_ord1 = sum_x1 / total
    avg_ord6 = sum_x6 / total
    avg_ord4 = sum_y / total

    print(f"평균값: ord1={avg_ord1:.1f}, ord4={avg_ord4:.1f}, ord6={avg_ord6:.1f}")

    # ord4 ≈ (ord1 + ord6) / 2 + offset 형태
    midpoint_errors = []
    for d in data:
        mid = (d['ord1'] + d['ord6']) / 2
        error = d['ord4'] - mid
        midpoint_errors.append(error)

    avg_offset = sum(midpoint_errors) / len(midpoint_errors)
    print(f"\nord4 ≈ (ord1 + ord6) / 2 + {avg_offset:.1f}")

    # 오차 분석
    errors_with_formula = [d['ord4'] - ((d['ord1'] + d['ord6']) / 2 + avg_offset) for d in data]
    mae = sum(abs(e) for e in errors_with_formula) / len(errors_with_formula)
    print(f"평균 절대 오차 (MAE): {mae:.1f}")

    # 더 정교한 공식: ord4 ≈ ord1 * a + ord6 * b
    # 단순히 비율로
    ratios = [(d['ord4'] - d['ord1']) / (d['ord6'] - d['ord1']) for d in data if d['ord6'] != d['ord1']]
    avg_ratio = sum(ratios) / len(ratios)
    print(f"\nord4 ≈ ord1 + (ord6 - ord1) * {avg_ratio:.3f}")

    # 이 공식의 오차
    errors_ratio = [d['ord4'] - (d['ord1'] + (d['ord6'] - d['ord1']) * avg_ratio) for d in data]
    mae_ratio = sum(abs(e) for e in errors_ratio) / len(errors_ratio)
    print(f"평균 절대 오차 (MAE): {mae_ratio:.1f}")

    # ========== 10. 핵심 발견 요약 ==========
    print("\n" + "=" * 80)
    print("[핵심 발견 요약]")
    print("=" * 80)
    print(f"""
1. ord4 범위: {min(ord4_values)}~{max(ord4_values)}, 평균 {sum(ord4_values)/len(ord4_values):.1f}
   → 주로 20-35 구간에 집중

2. 구간별: 20-29 ({range_freq.get(2,0)/total*100:.1f}%), 30-39 ({range_freq.get(3,0)/total*100:.1f}%)
   → 두 구간 합이 약 {(range_freq.get(2,0)+range_freq.get(3,0))/total*100:.0f}%

3. 상대 위치: 평균 {avg_pos:.3f} (이론 0.6)
   → ord4는 중앙보다 약간 뒤쪽에 위치

4. 예측 공식: ord4 ≈ ord1 + (ord6 - ord1) * {avg_ratio:.2f}
   → MAE = {mae_ratio:.1f}

5. 연속수: ord3-ord4 연속 {consec_with_ord3/total*100:.1f}%, ord4-ord5 연속 {consec_with_ord5/total*100:.1f}%
""")


def main():
    data = load_data()
    analyze(data)


if __name__ == "__main__":
    main()
