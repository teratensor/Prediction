"""
첫수끝수(FirstEnd) 인사이트 생성

당첨번호 중 첫번째(최소) 번호와 마지막(최대) 번호 분석
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"


def load_data() -> list:
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'first': balls[0],
                'end': balls[-1]
            })
    return results


def generate_insight():
    """첫수끝수 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # 첫수(최소) 분석
    first_freq = Counter(r['first'] for r in data)
    avg_first = sum(r['first'] for r in data) / total_rounds

    # 끝수(최대) 분석
    end_freq = Counter(r['end'] for r in data)
    avg_end = sum(r['end'] for r in data) / total_rounds

    # 범위(끝수-첫수) 분석
    span_freq = Counter(r['end'] - r['first'] for r in data)
    avg_span = sum(r['end'] - r['first'] for r in data) / total_rounds

    # 첫수-끝수 조합 분석
    pair_freq = Counter((r['first'], r['end']) for r in data)

    # first_distribution.csv 저장
    with open(OUTPUT_DIR / "first_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['value', 'frequency', 'ratio', 'probability'])
        for v in sorted(first_freq.keys()):
            writer.writerow([v, first_freq[v], round(first_freq[v]/total_rounds*100, 1), first_freq[v]/total_rounds])

    # end_distribution.csv 저장
    with open(OUTPUT_DIR / "end_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['value', 'frequency', 'ratio', 'probability'])
        for v in sorted(end_freq.keys()):
            writer.writerow([v, end_freq[v], round(end_freq[v]/total_rounds*100, 1), end_freq[v]/total_rounds])

    # span_distribution.csv 저장
    with open(OUTPUT_DIR / "span_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['span', 'frequency', 'ratio', 'probability'])
        for s in sorted(span_freq.keys()):
            writer.writerow([s, span_freq[s], round(span_freq[s]/total_rounds*100, 1), span_freq[s]/total_rounds])

    # pair_distribution.csv 저장
    with open(OUTPUT_DIR / "pair_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['first', 'end', 'frequency', 'ratio', 'probability'])
        for (first, end), c in pair_freq.most_common():
            writer.writerow([first, end, c, round(c/total_rounds*100, 1), c/total_rounds])

    # summary.csv 저장
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'category', 'value'])
        writer.writerow(['total_rounds', '', total_rounds])
        writer.writerow(['average', 'first', round(avg_first, 2)])
        writer.writerow(['min', 'first', min(r['first'] for r in data)])
        writer.writerow(['max', 'first', max(r['first'] for r in data)])
        writer.writerow(['average', 'end', round(avg_end, 2)])
        writer.writerow(['min', 'end', min(r['end'] for r in data)])
        writer.writerow(['max', 'end', max(r['end'] for r in data)])
        writer.writerow(['average', 'span', round(avg_span, 2)])
        writer.writerow(['min', 'span', min(r['end'] - r['first'] for r in data)])
        writer.writerow(['max', 'span', max(r['end'] - r['first'] for r in data)])

    print(f"=== 첫수끝수 인사이트 생성 완료 ===\n")
    print(f"총 {total_rounds}회차 분석")
    print()
    print("[첫수(최소값)]")
    print(f"  평균: {avg_first:.2f}")
    print(f"  가장 빈번: ", end="")
    for v, c in first_freq.most_common(5):
        print(f"{v}({c}회) ", end="")
    print()
    print()
    print("[끝수(최대값)]")
    print(f"  평균: {avg_end:.2f}")
    print(f"  가장 빈번: ", end="")
    for v, c in end_freq.most_common(5):
        print(f"{v}({c}회) ", end="")
    print()
    print()
    print("[범위(끝수-첫수)]")
    print(f"  평균: {avg_span:.2f}")
    print(f"  가장 빈번: ", end="")
    for s, c in span_freq.most_common(5):
        print(f"{s}({c}회) ", end="")
    print()
    print()
    print("CSV 파일 저장 완료")


if __name__ == "__main__":
    generate_insight()
