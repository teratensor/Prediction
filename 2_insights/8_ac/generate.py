"""
AC (Arithmetic Complexity) 값 분석

AC값: 6개 번호의 모든 쌍 차이값 중 유니크한 개수 - 5
- 6개 번호 → 15개 쌍 (6C2)
- 최소 AC: 0 (연속 6개 번호, 예: 1,2,3,4,5,6)
- 최대 AC: 10 (15개 차이값이 모두 다름)

예: (1,2,12,23,40,41)
차이값: 1,11,22,39,40,10,21,38,39,11,28,29,17,18,1
유니크: {1,10,11,17,18,21,22,28,29,38,39,40} = 12개
AC = 12 - 5 = 7
"""

import csv
from pathlib import Path
from collections import Counter
from itertools import combinations

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent.parent / "1_data" / "winning_numbers.csv"
STATS_DIR = BASE_DIR / "statistics"


def calculate_ac(balls):
    """AC값 계산: 모든 쌍의 차이값 유니크 개수 - 5"""
    differences = set()
    for b1, b2 in combinations(balls, 2):
        diff = abs(b2 - b1)
        differences.add(diff)
    return len(differences) - 5


def load_data():
    """당첨번호 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = tuple(sorted([int(row[f'ball{i}']) for i in range(1, 7)]))
            ac = calculate_ac(balls)
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'ac': ac,
            })
    return sorted(results, key=lambda x: x['round'])


def analyze_and_save(data):
    """AC값 분석 및 저장"""
    STATS_DIR.mkdir(exist_ok=True)

    total = len(data)
    print(f"총 {total}회차 분석")

    # AC값 분포
    ac_freq = Counter(d['ac'] for d in data)

    # 1. AC값 분포 저장
    with open(STATS_DIR / "ac_distribution.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ac', 'freq', 'pct'])
        for ac in range(11):  # 0~10
            freq = ac_freq.get(ac, 0)
            pct = freq / total * 100
            writer.writerow([ac, freq, f"{pct:.2f}"])

    # 2. 회차별 AC값 저장
    with open(STATS_DIR / "ac_by_round.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'balls', 'ac'])
        for d in data:
            balls_str = '-'.join(map(str, d['balls']))
            writer.writerow([d['round'], balls_str, d['ac']])

    # 3. 요약 통계
    ac_values = [d['ac'] for d in data]
    ac_avg = sum(ac_values) / len(ac_values)
    ac_min, ac_max = min(ac_values), max(ac_values)
    ac_mode = ac_freq.most_common(1)[0]

    # 구간별 집계
    low_ac = sum(1 for ac in ac_values if ac <= 4)  # 0-4
    mid_ac = sum(1 for ac in ac_values if 5 <= ac <= 7)  # 5-7
    high_ac = sum(1 for ac in ac_values if ac >= 8)  # 8-10

    with open(STATS_DIR / "summary.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['항목', '값'])
        writer.writerow(['총 회차', total])
        writer.writerow([''])
        writer.writerow(['AC 평균', f"{ac_avg:.2f}"])
        writer.writerow(['AC 범위', f"{ac_min}-{ac_max}"])
        writer.writerow(['AC 최빈값', f"{ac_mode[0]} ({ac_mode[1]}회, {ac_mode[1]/total*100:.1f}%)"])
        writer.writerow(['']
        )
        writer.writerow(['AC 0-4 (낮음)', f"{low_ac}회 ({low_ac/total*100:.1f}%)"])
        writer.writerow(['AC 5-7 (중간)', f"{mid_ac}회 ({mid_ac/total*100:.1f}%)"])
        writer.writerow(['AC 8-10 (높음)', f"{high_ac}회 ({high_ac/total*100:.1f}%)"])

    # 출력
    print("\n" + "=" * 60)
    print("[AC값 분포]")
    print("=" * 60)

    print(f"\n평균: {ac_avg:.2f}, 범위: {ac_min}-{ac_max}")
    print(f"최빈값: {ac_mode[0]} ({ac_mode[1]}회, {ac_mode[1]/total*100:.1f}%)")

    print("\n<AC값별 빈도>")
    for ac in range(11):
        freq = ac_freq.get(ac, 0)
        pct = freq / total * 100
        bar = "█" * int(pct / 2)
        print(f"  AC={ac:2d}: {freq:3d}회 ({pct:5.1f}%) {bar}")

    print("\n<구간별 집계>")
    print(f"  AC 0-4 (낮음): {low_ac:3d}회 ({low_ac/total*100:.1f}%)")
    print(f"  AC 5-7 (중간): {mid_ac:3d}회 ({mid_ac/total*100:.1f}%)")
    print(f"  AC 8-10 (높음): {high_ac:3d}회 ({high_ac/total*100:.1f}%)")

    # AC값별 샘플
    print("\n<AC값별 샘플>")
    for ac in range(11):
        samples = [d for d in data if d['ac'] == ac]
        if samples:
            sample = samples[-1]  # 가장 최근
            print(f"  AC={ac:2d}: {sample['round']}회차 {sample['balls']}")

    # 최근 20회차
    print("\n<최근 20회차>")
    for d in data[-20:]:
        print(f"  {d['round']}회차: {d['balls']} → AC={d['ac']}")

    print(f"\n저장 완료: {STATS_DIR}")


def main():
    data = load_data()
    analyze_and_save(data)


if __name__ == "__main__":
    main()
