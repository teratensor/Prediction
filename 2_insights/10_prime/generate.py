"""
소수(Prime) 인사이트 생성

당첨번호 중 소수 개수 분석
소수: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43 (14개)
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"

# 1-45 범위의 소수
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


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


def count_primes(balls: list) -> int:
    """당첨번호 중 소수 개수"""
    return sum(1 for b in balls if b in PRIMES)


def generate_insight():
    """소수 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # 각 회차별 소수 개수 계산
    prime_counts = []
    for row in data:
        cnt = count_primes(row['balls'])
        prime_counts.append({
            'round': row['round'],
            'prime_count': cnt,
            'primes': [b for b in row['balls'] if b in PRIMES]
        })

    # 소수 개수별 빈도
    count_freq = Counter(p['prime_count'] for p in prime_counts)

    # 평균 소수 개수
    avg_primes = sum(p['prime_count'] for p in prime_counts) / total_rounds

    # 각 소수별 출현 빈도
    prime_freq = Counter()
    for row in data:
        for b in row['balls']:
            if b in PRIMES:
                prime_freq[b] += 1

    # count_distribution.csv 저장
    with open(OUTPUT_DIR / "count_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['count', 'frequency', 'ratio', 'probability'])
        for k in sorted(count_freq.keys()):
            writer.writerow([k, count_freq[k], round(count_freq[k]/total_rounds*100, 1), count_freq[k]/total_rounds])

    # prime_frequency.csv 저장
    with open(OUTPUT_DIR / "prime_frequency.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prime', 'frequency', 'ratio', 'probability'])
        for p in sorted(PRIMES):
            writer.writerow([p, prime_freq[p], round(prime_freq[p]/total_rounds*100, 1), prime_freq[p]/(total_rounds*6)])

    # summary.csv 저장
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_rounds', total_rounds])
        writer.writerow(['average_primes', round(avg_primes, 2)])
        writer.writerow(['min_primes', min(p['prime_count'] for p in prime_counts)])
        writer.writerow(['max_primes', max(p['prime_count'] for p in prime_counts)])
        writer.writerow(['total_primes_in_range', len(PRIMES)])

    print(f"=== 소수 인사이트 생성 완료 ===\n")
    print(f"총 {total_rounds}회차 분석")
    print(f"평균 소수 개수: {avg_primes:.2f}개")
    print()
    print("[소수 개수별 분포]")
    for cnt in sorted(count_freq.keys()):
        freq = count_freq[cnt]
        ratio = freq / total_rounds * 100
        print(f"  {cnt}개: {freq}회 ({ratio:.1f}%)")
    print()
    print("CSV 파일 저장 완료")


if __name__ == "__main__":
    generate_insight()
