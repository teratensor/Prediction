"""
ord1, ord4, ord6 (첫수, 넷째수, 끝수) 조합 통계 분석

출력:
- statistics/triple_distribution.csv: (ord1, ord4, ord6) 빈도 분포
- statistics/ord1_ord4_gap.csv: ord1-ord4 간격 분포
- statistics/ord4_ord6_gap.csv: ord4-ord6 간격 분포
- statistics/ord1_distribution.csv: ord1 단독 분포
- statistics/ord4_distribution.csv: ord4 단독 분포
- statistics/ord6_distribution.csv: ord6 단독 분포
- statistics/summary.csv: 요약 통계
"""

import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent / "1_data" / "winning_numbers.csv"
STATS_DIR = BASE_DIR / "statistics"


def load_data():
    """당첨번호 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = tuple(sorted([int(row[f'ball{i}']) for i in range(1, 7)]))
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'ord1': balls[0],
                'ord4': balls[3],
                'ord6': balls[5],
            })
    return sorted(results, key=lambda x: x['round'])


def analyze_and_save(data):
    """통계 분석 및 저장"""
    STATS_DIR.mkdir(exist_ok=True)

    total = len(data)
    print(f"총 {total}회차 분석")

    # ========== 1. 개별 분포 ==========
    ord1_freq = Counter(d['ord1'] for d in data)
    ord4_freq = Counter(d['ord4'] for d in data)
    ord6_freq = Counter(d['ord6'] for d in data)

    # ord1 분포
    with open(STATS_DIR / "ord1_distribution.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'freq', 'pct'])
        for num in range(1, 46):
            freq = ord1_freq.get(num, 0)
            pct = freq / total * 100
            writer.writerow([num, freq, f"{pct:.2f}"])

    # ord4 분포
    with open(STATS_DIR / "ord4_distribution.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord4', 'freq', 'pct'])
        for num in range(1, 46):
            freq = ord4_freq.get(num, 0)
            pct = freq / total * 100
            writer.writerow([num, freq, f"{pct:.2f}"])

    # ord6 분포
    with open(STATS_DIR / "ord6_distribution.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord6', 'freq', 'pct'])
        for num in range(1, 46):
            freq = ord6_freq.get(num, 0)
            pct = freq / total * 100
            writer.writerow([num, freq, f"{pct:.2f}"])

    # ========== 2. 간격 분포 ==========
    gap14_freq = Counter(d['ord4'] - d['ord1'] for d in data)
    gap46_freq = Counter(d['ord6'] - d['ord4'] for d in data)

    # ord1-ord4 간격
    with open(STATS_DIR / "ord1_ord4_gap.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'freq', 'pct'])
        for gap in sorted(gap14_freq.keys()):
            freq = gap14_freq[gap]
            pct = freq / total * 100
            writer.writerow([gap, freq, f"{pct:.2f}"])

    # ord4-ord6 간격
    with open(STATS_DIR / "ord4_ord6_gap.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'freq', 'pct'])
        for gap in sorted(gap46_freq.keys()):
            freq = gap46_freq[gap]
            pct = freq / total * 100
            writer.writerow([gap, freq, f"{pct:.2f}"])

    # ========== 3. (ord1, ord4, ord6) 트리플 분포 ==========
    triple_freq = Counter((d['ord1'], d['ord4'], d['ord6']) for d in data)

    with open(STATS_DIR / "triple_distribution.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord4', 'ord6', 'freq', 'pct'])
        for (o1, o4, o6), freq in triple_freq.most_common():
            pct = freq / total * 100
            writer.writerow([o1, o4, o6, freq, f"{pct:.2f}"])

    # ========== 4. (ord1, ord6) 쌍 분포 (기존 firstend 참고) ==========
    pair16_freq = Counter((d['ord1'], d['ord6']) for d in data)

    with open(STATS_DIR / "ord1_ord6_pair.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord6', 'freq', 'pct'])
        for (o1, o6), freq in pair16_freq.most_common():
            pct = freq / total * 100
            writer.writerow([o1, o6, freq, f"{pct:.2f}"])

    # ========== 5. (ord1, ord4) 쌍 분포 ==========
    pair14_freq = Counter((d['ord1'], d['ord4']) for d in data)

    with open(STATS_DIR / "ord1_ord4_pair.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord4', 'freq', 'pct'])
        for (o1, o4), freq in pair14_freq.most_common():
            pct = freq / total * 100
            writer.writerow([o1, o4, freq, f"{pct:.2f}"])

    # ========== 6. (ord4, ord6) 쌍 분포 ==========
    pair46_freq = Counter((d['ord4'], d['ord6']) for d in data)

    with open(STATS_DIR / "ord4_ord6_pair.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord4', 'ord6', 'freq', 'pct'])
        for (o4, o6), freq in pair46_freq.most_common():
            pct = freq / total * 100
            writer.writerow([o4, o6, freq, f"{pct:.2f}"])

    # ========== 7. 요약 통계 ==========
    # ord1 통계
    ord1_values = [d['ord1'] for d in data]
    ord1_avg = sum(ord1_values) / len(ord1_values)
    ord1_min, ord1_max = min(ord1_values), max(ord1_values)
    ord1_mode = ord1_freq.most_common(1)[0]

    # ord4 통계
    ord4_values = [d['ord4'] for d in data]
    ord4_avg = sum(ord4_values) / len(ord4_values)
    ord4_min, ord4_max = min(ord4_values), max(ord4_values)
    ord4_mode = ord4_freq.most_common(1)[0]

    # ord6 통계
    ord6_values = [d['ord6'] for d in data]
    ord6_avg = sum(ord6_values) / len(ord6_values)
    ord6_min, ord6_max = min(ord6_values), max(ord6_values)
    ord6_mode = ord6_freq.most_common(1)[0]

    # 간격 통계
    gap14_values = [d['ord4'] - d['ord1'] for d in data]
    gap14_avg = sum(gap14_values) / len(gap14_values)
    gap14_mode = gap14_freq.most_common(1)[0]

    gap46_values = [d['ord6'] - d['ord4'] for d in data]
    gap46_avg = sum(gap46_values) / len(gap46_values)
    gap46_mode = gap46_freq.most_common(1)[0]

    with open(STATS_DIR / "summary.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['항목', '값'])
        writer.writerow(['총 회차', total])
        writer.writerow([''])
        writer.writerow(['ord1 평균', f"{ord1_avg:.1f}"])
        writer.writerow(['ord1 범위', f"{ord1_min}-{ord1_max}"])
        writer.writerow(['ord1 최빈값', f"{ord1_mode[0]} ({ord1_mode[1]}회)"])
        writer.writerow([''])
        writer.writerow(['ord4 평균', f"{ord4_avg:.1f}"])
        writer.writerow(['ord4 범위', f"{ord4_min}-{ord4_max}"])
        writer.writerow(['ord4 최빈값', f"{ord4_mode[0]} ({ord4_mode[1]}회)"])
        writer.writerow([''])
        writer.writerow(['ord6 평균', f"{ord6_avg:.1f}"])
        writer.writerow(['ord6 범위', f"{ord6_min}-{ord6_max}"])
        writer.writerow(['ord6 최빈값', f"{ord6_mode[0]} ({ord6_mode[1]}회)"])
        writer.writerow([''])
        writer.writerow(['ord1-ord4 간격 평균', f"{gap14_avg:.1f}"])
        writer.writerow(['ord1-ord4 간격 최빈', f"{gap14_mode[0]} ({gap14_mode[1]}회)"])
        writer.writerow(['ord4-ord6 간격 평균', f"{gap46_avg:.1f}"])
        writer.writerow(['ord4-ord6 간격 최빈', f"{gap46_mode[0]} ({gap46_mode[1]}회)"])
        writer.writerow([''])
        writer.writerow(['고유 트리플 수', len(triple_freq)])
        writer.writerow(['고유 (ord1,ord6) 쌍 수', len(pair16_freq)])
        writer.writerow(['고유 (ord1,ord4) 쌍 수', len(pair14_freq)])
        writer.writerow(['고유 (ord4,ord6) 쌍 수', len(pair46_freq)])

    # ========== 출력 ==========
    print("\n" + "=" * 60)
    print("[요약 통계]")
    print("=" * 60)

    print(f"\n<ord1 (첫수)>")
    print(f"  범위: {ord1_min}-{ord1_max}, 평균: {ord1_avg:.1f}")
    print(f"  최빈값: {ord1_mode[0]} ({ord1_mode[1]}회, {ord1_mode[1]/total*100:.1f}%)")
    print(f"  상위 5개: {[f'{n}({c}회)' for n, c in ord1_freq.most_common(5)]}")

    print(f"\n<ord4 (넷째수)>")
    print(f"  범위: {ord4_min}-{ord4_max}, 평균: {ord4_avg:.1f}")
    print(f"  최빈값: {ord4_mode[0]} ({ord4_mode[1]}회, {ord4_mode[1]/total*100:.1f}%)")
    print(f"  상위 5개: {[f'{n}({c}회)' for n, c in ord4_freq.most_common(5)]}")

    print(f"\n<ord6 (끝수)>")
    print(f"  범위: {ord6_min}-{ord6_max}, 평균: {ord6_avg:.1f}")
    print(f"  최빈값: {ord6_mode[0]} ({ord6_mode[1]}회, {ord6_mode[1]/total*100:.1f}%)")
    print(f"  상위 5개: {[f'{n}({c}회)' for n, c in ord6_freq.most_common(5)]}")

    print(f"\n<ord1-ord4 간격>")
    print(f"  평균: {gap14_avg:.1f}")
    print(f"  최빈값: {gap14_mode[0]} ({gap14_mode[1]}회)")
    print(f"  상위 5개: {[f'{g}({c}회)' for g, c in gap14_freq.most_common(5)]}")

    print(f"\n<ord4-ord6 간격>")
    print(f"  평균: {gap46_avg:.1f}")
    print(f"  최빈값: {gap46_mode[0]} ({gap46_mode[1]}회)")
    print(f"  상위 5개: {[f'{g}({c}회)' for g, c in gap46_freq.most_common(5)]}")

    print(f"\n<트리플 (ord1, ord4, ord6)>")
    print(f"  고유 조합 수: {len(triple_freq)}개")
    print(f"  상위 10개:")
    for (o1, o4, o6), freq in triple_freq.most_common(10):
        pct = freq / total * 100
        print(f"    ({o1:2d}, {o4:2d}, {o6:2d}): {freq}회 ({pct:.1f}%)")

    print(f"\n저장 완료: {STATS_DIR}")


def main():
    data = load_data()
    analyze_and_save(data)


if __name__ == "__main__":
    main()
