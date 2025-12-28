"""
Lastnum(끝수/6번째 번호) 인사이트 생성 - 전문가 분석

분석 항목:
1. ball6 기본 빈도 분석
2. ball6 구간별 분포 (0-4)
3. ball5-ball6 연속수 분석
4. ball6 소수 여부 분석
5. ball6 홀짝 분석
6. ball6 끝자리(0-9) 분석
7. ball1~5 합계와 ball6 관계
8. 범위(span = ball6 - ball1) 분석
9. 복합 패턴 분석
10. 최근 트렌드 분석
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import statistics

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"

# 소수 목록 (1~45)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


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
                'ball1': balls[0],
                'ball2': balls[1],
                'ball3': balls[2],
                'ball4': balls[3],
                'ball5': balls[4],
                'ball6': balls[5],
            })
    return results


def get_range_label(n: int) -> str:
    """구간 라벨 (0-4 세그먼트)"""
    if n <= 9: return '0(01-09)'
    elif n <= 19: return '1(10-19)'
    elif n <= 29: return '2(20-29)'
    elif n <= 39: return '3(30-39)'
    else: return '4(40-45)'


def is_prime(n: int) -> bool:
    return n in PRIMES


def generate_insight():
    """Lastnum 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # ========== 1. ball6 기본 빈도 분석 ==========
    ball6_freq = Counter(r['ball6'] for r in data)

    with open(OUTPUT_DIR / "ball6_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball6', 'frequency', 'ratio', 'probability', 'is_prime', 'rank'])
        sorted_items = sorted(ball6_freq.items(), key=lambda x: -x[1])
        for rank, (v, freq) in enumerate(sorted_items, 1):
            writer.writerow([v, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), is_prime(v), rank])

    # ========== 2. ball6 구간별 분포 ==========
    range_freq = Counter(get_range_label(r['ball6']) for r in data)

    with open(OUTPUT_DIR / "ball6_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'frequency', 'ratio', 'probability'])
        for rng in ['0(01-09)', '1(10-19)', '2(20-29)', '3(30-39)', '4(40-45)']:
            freq = range_freq.get(rng, 0)
            writer.writerow([rng, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 3. ball5-ball6 연속수 분석 ==========
    consecutive_freq = Counter()
    gap_freq = Counter()
    for r in data:
        gap = r['ball6'] - r['ball5']
        gap_freq[gap] += 1
        consecutive_freq[gap == 1] += 1

    with open(OUTPUT_DIR / "ball5_ball6_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'probability', 'is_consecutive'])
        for gap in sorted(gap_freq.keys()):
            freq = gap_freq[gap]
            writer.writerow([gap, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), gap == 1])

    # ========== 4. ball6 소수 여부 분석 ==========
    ball6_prime_freq = Counter(is_prime(r['ball6']) for r in data)

    with open(OUTPUT_DIR / "ball6_prime_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['is_prime', 'frequency', 'ratio', 'probability'])
        for is_p in [True, False]:
            freq = ball6_prime_freq.get(is_p, 0)
            label = '소수' if is_p else '비소수'
            writer.writerow([label, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 5. ball6 홀짝 분석 ==========
    oddeven_freq = Counter('홀' if r['ball6'] % 2 == 1 else '짝' for r in data)

    with open(OUTPUT_DIR / "ball6_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'frequency', 'ratio', 'probability'])
        for t in ['홀', '짝']:
            freq = oddeven_freq.get(t, 0)
            writer.writerow([t, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 6. ball6 끝자리 분석 ==========
    lastdigit_freq = Counter(r['ball6'] % 10 for r in data)

    with open(OUTPUT_DIR / "ball6_lastdigit_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['last_digit', 'frequency', 'ratio', 'probability'])
        for d in range(10):
            freq = lastdigit_freq.get(d, 0)
            writer.writerow([d, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 7. ball1~5 합계와 ball6 관계 ==========
    sum15_ball6 = defaultdict(list)
    for r in data:
        sum15 = sum(r['balls'][:5])
        # 10단위 구간
        range_key = (sum15 // 20) * 20
        sum15_ball6[range_key].append(r['ball6'])

    with open(OUTPUT_DIR / "sum15_ball6_relation.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sum15_range', 'count', 'ball6_avg', 'ball6_min', 'ball6_max', 'ball6_std'])
        for range_key in sorted(sum15_ball6.keys()):
            balls = sum15_ball6[range_key]
            if len(balls) >= 3:
                writer.writerow([f"{range_key}-{range_key+19}", len(balls),
                               round(statistics.mean(balls), 1), min(balls), max(balls),
                               round(statistics.stdev(balls), 2)])

    # ========== 8. 범위(span) 분석 ==========
    span_freq = Counter(r['ball6'] - r['ball1'] for r in data)

    with open(OUTPUT_DIR / "span_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['span', 'frequency', 'ratio', 'probability'])
        for span in sorted(span_freq.keys()):
            freq = span_freq[span]
            writer.writerow([span, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 9. 복합 패턴 분석 ==========
    complex_freq = Counter()
    for r in data:
        range_label = get_range_label(r['ball6'])
        is_consec = r['ball6'] - r['ball5'] == 1
        ball6_prime = is_prime(r['ball6'])
        oddeven = '홀' if r['ball6'] % 2 == 1 else '짝'
        key = (range_label, is_consec, ball6_prime, oddeven)
        complex_freq[key] += 1

    with open(OUTPUT_DIR / "complex_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'is_consecutive', 'is_prime', 'oddeven', 'frequency', 'ratio'])
        for (rng, consec, prime, oe), freq in complex_freq.most_common():
            writer.writerow([rng, consec, prime, oe, freq, round(freq/total_rounds*100, 1)])

    # ========== 10. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_freq = Counter(r['ball6'] for r in recent_50)
    recent_100_freq = Counter(r['ball6'] for r in recent_100)

    with open(OUTPUT_DIR / "ball6_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball6', 'all_time_freq', 'all_time_ratio', 'recent_100_freq',
                        'recent_100_ratio', 'recent_50_freq', 'recent_50_ratio', 'trend'])
        for v in sorted(ball6_freq.keys()):
            all_freq = ball6_freq[v]
            r100_freq = recent_100_freq.get(v, 0)
            r50_freq = recent_50_freq.get(v, 0)

            all_ratio = all_freq / total_rounds
            r100_ratio = r100_freq / 100 if r100_freq else 0
            r50_ratio = r50_freq / 50 if r50_freq else 0

            if r50_ratio > all_ratio * 1.3:
                trend = '상승'
            elif r50_ratio < all_ratio * 0.7:
                trend = '하락'
            else:
                trend = '보합'

            writer.writerow([v, all_freq, round(all_ratio*100, 1), r100_freq,
                           round(r100_ratio*100, 1), r50_freq, round(r50_ratio*100, 1), trend])

    # ========== Summary ==========
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['ball6_avg', round(statistics.mean(r['ball6'] for r in data), 2), 'ball6 평균'])
        writer.writerow(['ball6_std', round(statistics.stdev(r['ball6'] for r in data), 2), 'ball6 표준편차'])
        writer.writerow(['ball6_median', statistics.median(r['ball6'] for r in data), 'ball6 중앙값'])
        writer.writerow(['ball6_mode', ball6_freq.most_common(1)[0][0], 'ball6 최빈값'])
        writer.writerow(['ball6_min', min(r['ball6'] for r in data), 'ball6 최소값'])
        writer.writerow(['ball6_max', max(r['ball6'] for r in data), 'ball6 최대값'])
        writer.writerow(['consecutive_ratio', round(consecutive_freq[True]/total_rounds*100, 1), 'ball5-ball6 연속수 비율(%)'])
        writer.writerow(['ball6_prime_ratio', round(ball6_prime_freq[True]/total_rounds*100, 1), 'ball6 소수 비율(%)'])
        writer.writerow(['ball6_odd_ratio', round(oddeven_freq['홀']/total_rounds*100, 1), 'ball6 홀수 비율(%)'])
        writer.writerow(['range_4_ratio', round(range_freq['4(40-45)']/total_rounds*100, 1), 'ball6 구간4(40-45) 비율(%)'])
        writer.writerow(['avg_span', round(statistics.mean(r['ball6'] - r['ball1'] for r in data), 2), '평균 범위(span)'])

    # ========== 출력 ==========
    print("=" * 60)
    print("Lastnum(6번째 번호/끝수) 전문가 인사이트")
    print("=" * 60)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. ball6 기본 통계]")
    print(f"    평균: {statistics.mean(r['ball6'] for r in data):.1f}")
    print(f"    표준편차: {statistics.stdev(r['ball6'] for r in data):.1f}")
    print(f"    중앙값: {statistics.median(r['ball6'] for r in data)}")
    print(f"    최빈값: {ball6_freq.most_common(1)[0][0]} ({ball6_freq.most_common(1)[0][1]}회)")

    print("\n[2. ball6 빈도 상위 10개]")
    for v, c in ball6_freq.most_common(10):
        prime_mark = "★" if is_prime(v) else ""
        print(f"    {v:2d}{prime_mark}: {c:3d}회 ({c/total_rounds*100:5.1f}%)")

    print("\n[3. ball6 구간별 분포]")
    for rng in ['0(01-09)', '1(10-19)', '2(20-29)', '3(30-39)', '4(40-45)']:
        freq = range_freq.get(rng, 0)
        bar = '█' * int(freq/total_rounds*50)
        print(f"    {rng}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[4. ball5-ball6 간격 분포 (상위 5개)]")
    for gap, freq in sorted(gap_freq.items(), key=lambda x: -x[1])[:5]:
        label = "(연속)" if gap == 1 else ""
        print(f"    간격 {gap:2d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {label}")

    print("\n[5. ball6 소수 여부]")
    for is_p in [True, False]:
        freq = ball6_prime_freq.get(is_p, 0)
        label = '소수' if is_p else '비소수'
        print(f"    {label}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[6. ball6 홀짝 분포]")
    for t in ['홀', '짝']:
        freq = oddeven_freq.get(t, 0)
        print(f"    {t}수: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[7. 범위(span) 분포 상위 5개]")
    for span, freq in sorted(span_freq.items(), key=lambda x: -x[1])[:5]:
        print(f"    범위 {span:2d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    avg_span = statistics.mean(r['ball6'] - r['ball1'] for r in data)
    print(f"\n    평균 범위: {avg_span:.1f}")

    print("\n[8. 최근 트렌드 (상승 번호)]")
    trending_up = []
    for v in ball6_freq.keys():
        all_ratio = ball6_freq[v] / total_rounds
        r50_ratio = recent_50_freq.get(v, 0) / 50
        if r50_ratio > all_ratio * 1.3:
            trending_up.append((v, r50_ratio * 100, all_ratio * 100))
    for v, r50, all_r in sorted(trending_up, key=lambda x: -x[1])[:5]:
        prime_mark = "★" if is_prime(v) else ""
        print(f"    {v:2d}{prime_mark}: 최근50회 {r50:.1f}% (전체 {all_r:.1f}%) ↑")

    print("\n" + "=" * 60)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_insight()
