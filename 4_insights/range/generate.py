"""
구간(Range) 인사이트 생성

당첨번호를 구간별로 분류하여 분석
구간: 1-10, 11-20, 21-30, 31-40, 41-45
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"

# 구간 정의
RANGES = [
    (1, 10, "1-10"),
    (11, 20, "11-20"),
    (21, 30, "21-30"),
    (31, 40, "31-40"),
    (41, 45, "41-45")
]


def load_data() -> list:
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = [int(row[f'ball{i}']) for i in range(1, 7)]
            results.append({
                'round': int(row['round']),
                'balls': balls
            })
    return results


def get_range(ball: int) -> str:
    """번호가 속한 구간 반환"""
    for start, end, label in RANGES:
        if start <= ball <= end:
            return label
    return "unknown"


def get_range_counts(balls: list) -> dict:
    """각 구간별 번호 개수"""
    counts = {label: 0 for _, _, label in RANGES}
    for b in balls:
        r = get_range(b)
        if r in counts:
            counts[r] += 1
    return counts


def get_range_code(balls: list) -> str:
    """구간 코드 생성 (예: 21210)"""
    counts = get_range_counts(balls)
    return ''.join(str(counts[label]) for _, _, label in RANGES)


def generate_insight():
    """구간 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # 각 회차별 구간 분석
    range_results = []
    for row in data:
        counts = get_range_counts(row['balls'])
        code = get_range_code(row['balls'])
        range_results.append({
            'round': row['round'],
            'counts': counts,
            'code': code
        })

    # 구간 코드별 빈도
    code_freq = Counter(r['code'] for r in range_results)

    # 각 구간별 평균 개수
    avg_counts = {}
    for _, _, label in RANGES:
        total = sum(r['counts'][label] for r in range_results)
        avg_counts[label] = total / total_rounds

    # 각 구간별 개수 분포
    range_distributions = {}
    for _, _, label in RANGES:
        dist = Counter(r['counts'][label] for r in range_results)
        range_distributions[label] = dict(sorted(dist.items()))

    # pattern_distribution.csv 저장
    with open(OUTPUT_DIR / "pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'frequency', 'ratio', 'probability'])
        for c, n in code_freq.most_common():
            writer.writerow([c, n, round(n/total_rounds*100, 1), n/total_rounds])

    # range_count_distribution.csv 저장 (각 구간별 개수 분포)
    with open(OUTPUT_DIR / "range_count_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'count', 'frequency', 'ratio', 'probability'])
        for label, dist in range_distributions.items():
            for k, v in sorted(dist.items()):
                writer.writerow([label, k, v, round(v/total_rounds*100, 1), v/total_rounds])

    # summary.csv 저장
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'range', 'value'])
        writer.writerow(['total_rounds', '', total_rounds])
        writer.writerow(['unique_patterns', '', len(code_freq)])
        for _, _, label in RANGES:
            writer.writerow(['average', label, round(avg_counts[label], 2)])

    print(f"=== 구간 인사이트 생성 완료 ===\n")
    print(f"총 {total_rounds}회차 분석")
    print(f"고유 패턴: {len(code_freq)}개")
    print()
    print("[구간별 평균 개수]")
    for _, _, label in RANGES:
        print(f"  {label}: {avg_counts[label]:.2f}개")
    print()
    print("[가장 빈번한 패턴 Top 10]")
    for code, cnt in code_freq.most_common(10):
        ratio = cnt / total_rounds * 100
        print(f"  {code}: {cnt}회 ({ratio:.1f}%)")
    print()
    print("CSV 파일 저장 완료")


if __name__ == "__main__":
    generate_insight()
