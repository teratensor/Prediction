"""
firstend 인사이트 적중률 분석

목표: ball1(첫수)과 ball6(끝수)가 각각 최빈 구간에 들어오는 비율 확인
- ball1: 1-10 (79%)
- ball6: 38-45 (57.8%) 또는 40-45 (더 좁은 범위)
- 둘 다 적중: 88.1%?
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"


def load_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
            })
    return results


def analyze_firstend(data, start_round=900):
    """firstend 적중률 분석"""

    # 시작 회차 찾기
    start_idx = None
    for i, d in enumerate(data):
        if d['round'] >= start_round:
            start_idx = i
            break

    if start_idx is None:
        start_idx = 0

    results = data[start_idx:]
    total = len(results)

    print("=" * 70)
    print(f"firstend 인사이트 적중률 분석 ({data[start_idx]['round']}~{data[-1]['round']}회차, {total}회)")
    print("=" * 70)

    # ball1 (첫수) 분석
    print("\n[ball1 (첫수) 분석]")
    ball1_ranges = {
        '1-5': (1, 5),
        '1-10': (1, 10),
        '1-15': (1, 15),
    }

    for name, (lo, hi) in ball1_ranges.items():
        hit = sum(1 for r in results if lo <= r['balls'][0] <= hi)
        print(f"  {name}: {hit}/{total} ({hit/total*100:.1f}%)")

    # ball1 분포
    ball1_dist = Counter(r['balls'][0] for r in results)
    print(f"\n  실제 분포 (상위 10개):")
    for num, cnt in ball1_dist.most_common(10):
        print(f"    {num:2d}: {cnt}회 ({cnt/total*100:.1f}%)")

    # ball6 (끝수) 분석
    print("\n[ball6 (끝수) 분석]")
    ball6_ranges = {
        '40-45': (40, 45),
        '38-45': (38, 45),
        '35-45': (35, 45),
    }

    for name, (lo, hi) in ball6_ranges.items():
        hit = sum(1 for r in results if lo <= r['balls'][5] <= hi)
        print(f"  {name}: {hit}/{total} ({hit/total*100:.1f}%)")

    # ball6 분포
    ball6_dist = Counter(r['balls'][5] for r in results)
    print(f"\n  실제 분포 (상위 10개):")
    for num, cnt in ball6_dist.most_common(10):
        print(f"    {num:2d}: {cnt}회 ({cnt/total*100:.1f}%)")

    # 조합 분석 (ball1 AND ball6)
    print("\n[firstend 조합 적중률 (ball1 AND ball6)]")
    combinations = [
        ('ball1: 1-10, ball6: 40-45', (1, 10), (40, 45)),
        ('ball1: 1-10, ball6: 38-45', (1, 10), (38, 45)),
        ('ball1: 1-15, ball6: 38-45', (1, 15), (38, 45)),
        ('ball1: 1-15, ball6: 35-45', (1, 15), (35, 45)),
    ]

    for name, (b1_lo, b1_hi), (b6_lo, b6_hi) in combinations:
        hit = sum(1 for r in results
                  if b1_lo <= r['balls'][0] <= b1_hi and b6_lo <= r['balls'][5] <= b6_hi)
        print(f"  {name}: {hit}/{total} ({hit/total*100:.1f}%)")

    # OR 조건 (둘 중 하나라도)
    print("\n[firstend OR 조건 (ball1 OR ball6)]")
    for name, (b1_lo, b1_hi), (b6_lo, b6_hi) in combinations:
        hit = sum(1 for r in results
                  if (b1_lo <= r['balls'][0] <= b1_hi) or (b6_lo <= r['balls'][5] <= b6_hi))
        print(f"  {name}: {hit}/{total} ({hit/total*100:.1f}%)")

    # 각각 개별 적중률
    print("\n[개별 적중률 요약]")
    b1_1_10 = sum(1 for r in results if 1 <= r['balls'][0] <= 10)
    b6_40_45 = sum(1 for r in results if 40 <= r['balls'][5] <= 45)
    b6_38_45 = sum(1 for r in results if 38 <= r['balls'][5] <= 45)

    print(f"  ball1 (1-10): {b1_1_10}/{total} ({b1_1_10/total*100:.1f}%)")
    print(f"  ball6 (40-45): {b6_40_45}/{total} ({b6_40_45/total*100:.1f}%)")
    print(f"  ball6 (38-45): {b6_38_45}/{total} ({b6_38_45/total*100:.1f}%)")

    # 둘 다 적중
    both_hit = sum(1 for r in results
                   if 1 <= r['balls'][0] <= 10 and 38 <= r['balls'][5] <= 45)
    print(f"\n  둘 다 적중 (1-10 AND 38-45): {both_hit}/{total} ({both_hit/total*100:.1f}%)")

    # 이론적 확률과 비교
    print("\n[이론적 확률 vs 실제]")
    # 독립 가정시: P(ball1 in 1-10) * P(ball6 in 38-45)
    p_b1 = b1_1_10 / total
    p_b6 = b6_38_45 / total
    theoretical = p_b1 * p_b6
    actual = both_hit / total
    print(f"  이론적 (독립 가정): {theoretical*100:.1f}%")
    print(f"  실제: {actual*100:.1f}%")
    print(f"  차이: {(actual - theoretical)*100:+.1f}%p")


def analyze_all_rounds(data):
    """전체 회차 분석"""
    total = len(data)

    print("\n" + "=" * 70)
    print(f"전체 회차 분석 (1~{data[-1]['round']}회차, {total}회)")
    print("=" * 70)

    # ball1
    b1_1_10 = sum(1 for r in data if 1 <= r['balls'][0] <= 10)
    b1_1_15 = sum(1 for r in data if 1 <= r['balls'][0] <= 15)

    # ball6
    b6_40_45 = sum(1 for r in data if 40 <= r['balls'][5] <= 45)
    b6_38_45 = sum(1 for r in data if 38 <= r['balls'][5] <= 45)
    b6_35_45 = sum(1 for r in data if 35 <= r['balls'][5] <= 45)

    # 조합
    both_1_10_40_45 = sum(1 for r in data if 1 <= r['balls'][0] <= 10 and 40 <= r['balls'][5] <= 45)
    both_1_10_38_45 = sum(1 for r in data if 1 <= r['balls'][0] <= 10 and 38 <= r['balls'][5] <= 45)

    print(f"\n[ball1 적중률]")
    print(f"  1-10: {b1_1_10}/{total} ({b1_1_10/total*100:.1f}%)")
    print(f"  1-15: {b1_1_15}/{total} ({b1_1_15/total*100:.1f}%)")

    print(f"\n[ball6 적중률]")
    print(f"  40-45: {b6_40_45}/{total} ({b6_40_45/total*100:.1f}%)")
    print(f"  38-45: {b6_38_45}/{total} ({b6_38_45/total*100:.1f}%)")
    print(f"  35-45: {b6_35_45}/{total} ({b6_35_45/total*100:.1f}%)")

    print(f"\n[둘 다 적중 (AND)]")
    print(f"  ball1(1-10) AND ball6(40-45): {both_1_10_40_45}/{total} ({both_1_10_40_45/total*100:.1f}%)")
    print(f"  ball1(1-10) AND ball6(38-45): {both_1_10_38_45}/{total} ({both_1_10_38_45/total*100:.1f}%)")

    # OR 조건
    either_1 = sum(1 for r in data if (1 <= r['balls'][0] <= 10) or (40 <= r['balls'][5] <= 45))
    either_2 = sum(1 for r in data if (1 <= r['balls'][0] <= 10) or (38 <= r['balls'][5] <= 45))

    print(f"\n[둘 중 하나라도 (OR)]")
    print(f"  ball1(1-10) OR ball6(40-45): {either_1}/{total} ({either_1/total*100:.1f}%)")
    print(f"  ball1(1-10) OR ball6(38-45): {either_2}/{total} ({either_2/total*100:.1f}%)")


def main():
    data = load_data()
    print(f"총 데이터: {len(data)}회차\n")

    # 900회차부터 분석
    analyze_firstend(data, start_round=900)

    # 전체 회차 분석
    analyze_all_rounds(data)


if __name__ == "__main__":
    main()
