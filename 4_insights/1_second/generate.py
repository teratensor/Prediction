"""
Second(2번째 번호) 인사이트 생성 - 전문가 분석

분석 항목:
1. ball2 기본 빈도 분석
2. ball2 구간별 분포 (0: 1-9, 1: 10-19, 2: 20-29, 3: 30-39, 4: 40-45)
3. ball1 소수 여부별 ball2 분석
4. ball1-ball2 연속수 분석
5. ball2 자신의 소수 여부 분석
6. ball2 홀짝 분석
7. ball2 끝자리(0-9) 분석
8. ball1별 ball2 분포 분석
9. ball1-ball2 간격별 ball3 예측 관계
10. 복합 패턴 분석 (구간 + 소수 + 홀짝)
11. 최근 N회 트렌드 분석
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


def get_range_num(n: int) -> int:
    """구간 번호"""
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def is_prime(n: int) -> bool:
    return n in PRIMES


def generate_insight():
    """Second 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # ========== 1. ball2 기본 빈도 분석 ==========
    ball2_freq = Counter(r['ball2'] for r in data)

    with open(OUTPUT_DIR / "ball2_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball2', 'frequency', 'ratio', 'probability', 'is_prime', 'rank'])
        sorted_items = sorted(ball2_freq.items(), key=lambda x: -x[1])
        for rank, (v, freq) in enumerate(sorted_items, 1):
            writer.writerow([v, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), is_prime(v), rank])

    # ========== 2. ball2 구간별 분포 ==========
    range_freq = Counter(get_range_label(r['ball2']) for r in data)

    with open(OUTPUT_DIR / "ball2_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'frequency', 'ratio', 'probability'])
        for rng in ['0(01-09)', '1(10-19)', '2(20-29)', '3(30-39)', '4(40-45)']:
            freq = range_freq.get(rng, 0)
            writer.writerow([rng, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 3. ball1 소수 여부별 ball2 분석 ==========
    ball1_prime_ball2 = defaultdict(list)
    for r in data:
        key = 'ball1_소수' if is_prime(r['ball1']) else 'ball1_비소수'
        ball1_prime_ball2[key].append(r['ball2'])

    with open(OUTPUT_DIR / "ball1_prime_ball2.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball1_type', 'count', 'ball2_avg', 'ball2_std', 'ball2_min', 'ball2_max',
                        'ball2_prime_count', 'ball2_prime_ratio'])
        for key in ['ball1_소수', 'ball1_비소수']:
            balls = ball1_prime_ball2[key]
            prime_count = sum(1 for b in balls if is_prime(b))
            writer.writerow([key, len(balls), round(statistics.mean(balls), 1),
                           round(statistics.stdev(balls), 2), min(balls), max(balls),
                           prime_count, round(prime_count/len(balls)*100, 1)])

    # ========== 4. ball1-ball2 연속수 분석 ==========
    consecutive_freq = Counter()
    gap_freq = Counter()
    for r in data:
        gap = r['ball2'] - r['ball1']
        gap_freq[gap] += 1
        consecutive_freq[gap == 1] += 1

    with open(OUTPUT_DIR / "ball1_ball2_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'probability', 'is_consecutive'])
        for gap in sorted(gap_freq.keys()):
            freq = gap_freq[gap]
            writer.writerow([gap, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), gap == 1])

    # ========== 5. ball2 소수 여부 분석 ==========
    ball2_prime_freq = Counter(is_prime(r['ball2']) for r in data)

    with open(OUTPUT_DIR / "ball2_prime_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['is_prime', 'frequency', 'ratio', 'probability'])
        for is_p in [True, False]:
            freq = ball2_prime_freq.get(is_p, 0)
            label = '소수' if is_p else '비소수'
            writer.writerow([label, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 6. ball2 홀짝 분석 ==========
    oddeven_freq = Counter('홀' if r['ball2'] % 2 == 1 else '짝' for r in data)

    with open(OUTPUT_DIR / "ball2_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'frequency', 'ratio', 'probability'])
        for t in ['홀', '짝']:
            freq = oddeven_freq.get(t, 0)
            writer.writerow([t, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 7. ball2 끝자리 분석 ==========
    lastdigit_freq = Counter(r['ball2'] % 10 for r in data)

    with open(OUTPUT_DIR / "ball2_lastdigit_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['last_digit', 'frequency', 'ratio', 'probability'])
        for d in range(10):
            freq = lastdigit_freq.get(d, 0)
            writer.writerow([d, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 8. ball1별 ball2 분석 ==========
    ball1_ball2 = defaultdict(list)
    for r in data:
        ball1_ball2[r['ball1']].append(r['ball2'])

    with open(OUTPUT_DIR / "ball1_ball2_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball1', 'count', 'ball2_avg', 'ball2_min', 'ball2_max', 'ball2_mode',
                        'ball1_is_prime', 'consecutive_count', 'consecutive_ratio'])
        for b1 in sorted(ball1_ball2.keys()):
            balls = ball1_ball2[b1]
            if len(balls) >= 3:
                mode = Counter(balls).most_common(1)[0][0]
                consec_count = sum(1 for b2 in balls if b2 == b1 + 1)
                writer.writerow([b1, len(balls), round(statistics.mean(balls), 1),
                               min(balls), max(balls), mode, is_prime(b1),
                               consec_count, round(consec_count/len(balls)*100, 1)])

    # ========== 9. ball1-ball2 간격별 ball3 관계 ==========
    gap12_ball3 = defaultdict(list)
    for r in data:
        gap = r['ball2'] - r['ball1']
        gap12_ball3[gap].append(r['ball3'])

    with open(OUTPUT_DIR / "gap12_ball3_relation.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball1_ball2_gap', 'count', 'ball3_avg', 'ball3_min', 'ball3_max', 'ball3_std'])
        for gap in sorted(gap12_ball3.keys()):
            balls = gap12_ball3[gap]
            if len(balls) >= 3:
                writer.writerow([gap, len(balls), round(statistics.mean(balls), 1),
                               min(balls), max(balls), round(statistics.stdev(balls), 2)])

    # ========== 10. 복합 패턴 분석 ==========
    complex_freq = Counter()
    for r in data:
        range_label = get_range_num(r['ball2'])
        is_consec = r['ball2'] - r['ball1'] == 1
        ball2_prime = is_prime(r['ball2'])
        oddeven = '홀' if r['ball2'] % 2 == 1 else '짝'
        key = (range_label, is_consec, ball2_prime, oddeven)
        complex_freq[key] += 1

    with open(OUTPUT_DIR / "complex_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'is_consecutive', 'is_prime', 'oddeven', 'frequency', 'ratio', 'probability'])
        for (rng, consec, prime, oe), freq in complex_freq.most_common():
            writer.writerow([rng, consec, prime, oe, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 11. ball1, ball2 동시 소수 패턴 ==========
    prime_pattern_freq = Counter()
    for r in data:
        p1 = 'P' if is_prime(r['ball1']) else 'N'
        p2 = 'P' if is_prime(r['ball2']) else 'N'
        prime_pattern_freq[p1 + p2] += 1

    with open(OUTPUT_DIR / "ball1_ball2_prime_pattern.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern', 'ball1_prime', 'ball2_prime', 'frequency', 'ratio', 'probability'])
        for pattern in ['PP', 'PN', 'NP', 'NN']:
            freq = prime_pattern_freq.get(pattern, 0)
            writer.writerow([pattern, pattern[0] == 'P', pattern[1] == 'P', freq,
                           round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 12. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_freq = Counter(r['ball2'] for r in recent_50)
    recent_100_freq = Counter(r['ball2'] for r in recent_100)

    with open(OUTPUT_DIR / "ball2_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball2', 'all_time_freq', 'all_time_ratio', 'recent_100_freq',
                        'recent_100_ratio', 'recent_50_freq', 'recent_50_ratio', 'trend'])
        for v in sorted(ball2_freq.keys()):
            all_freq = ball2_freq[v]
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
        writer.writerow(['ball2_avg', round(statistics.mean(r['ball2'] for r in data), 2), 'ball2 평균'])
        writer.writerow(['ball2_std', round(statistics.stdev(r['ball2'] for r in data), 2), 'ball2 표준편차'])
        writer.writerow(['ball2_median', statistics.median(r['ball2'] for r in data), 'ball2 중앙값'])
        writer.writerow(['ball2_mode', ball2_freq.most_common(1)[0][0], 'ball2 최빈값'])
        writer.writerow(['ball2_min', min(r['ball2'] for r in data), 'ball2 최소값'])
        writer.writerow(['ball2_max', max(r['ball2'] for r in data), 'ball2 최대값'])
        writer.writerow(['consecutive_ratio', round(consecutive_freq[True]/total_rounds*100, 1), 'ball1-ball2 연속수 비율(%)'])
        writer.writerow(['ball2_prime_ratio', round(ball2_prime_freq[True]/total_rounds*100, 1), 'ball2 소수 비율(%)'])
        writer.writerow(['ball2_odd_ratio', round(oddeven_freq['홀']/total_rounds*100, 1), 'ball2 홀수 비율(%)'])
        writer.writerow(['range_0_ratio', round(range_freq['0(01-09)']/total_rounds*100, 1), 'ball2 구간0(01-09) 비율(%)'])

    # ========== 출력 ==========
    print("=" * 60)
    print("Second(2번째 번호) 전문가 인사이트")
    print("=" * 60)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. ball2 기본 통계]")
    print(f"    평균: {statistics.mean(r['ball2'] for r in data):.1f}")
    print(f"    표준편차: {statistics.stdev(r['ball2'] for r in data):.1f}")
    print(f"    중앙값: {statistics.median(r['ball2'] for r in data)}")
    print(f"    최빈값: {ball2_freq.most_common(1)[0][0]} ({ball2_freq.most_common(1)[0][1]}회)")

    print("\n[2. ball2 빈도 상위 10개]")
    for v, c in ball2_freq.most_common(10):
        prime_mark = "★" if is_prime(v) else ""
        print(f"    {v:2d}{prime_mark}: {c:3d}회 ({c/total_rounds*100:5.1f}%)")

    print("\n[3. ball2 구간별 분포]")
    for rng in ['0(01-09)', '1(10-19)', '2(20-29)', '3(30-39)', '4(40-45)']:
        freq = range_freq.get(rng, 0)
        bar = '█' * int(freq/total_rounds*50)
        print(f"    {rng}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[4. ball1-ball2 간격 분포 (상위 5개)]")
    for gap, freq in sorted(gap_freq.items(), key=lambda x: -x[1])[:5]:
        label = "(연속)" if gap == 1 else ""
        print(f"    간격 {gap:2d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {label}")

    print("\n[5. ball1 소수 여부별 ball2]")
    for key in ['ball1_소수', 'ball1_비소수']:
        balls = ball1_prime_ball2[key]
        prime_count = sum(1 for b in balls if is_prime(b))
        print(f"    {key}: {len(balls):3d}회, ball2 평균={statistics.mean(balls):.1f}, ball2 소수비율={prime_count/len(balls)*100:.1f}%")

    print("\n[6. ball2 소수 여부]")
    for is_p in [True, False]:
        freq = ball2_prime_freq.get(is_p, 0)
        label = '소수' if is_p else '비소수'
        print(f"    {label}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[7. ball2 홀짝 분포]")
    for t in ['홀', '짝']:
        freq = oddeven_freq.get(t, 0)
        print(f"    {t}수: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[8. ball2 끝자리 분포]")
    for d in range(10):
        freq = lastdigit_freq.get(d, 0)
        bar = '█' * int(freq/total_rounds*30)
        print(f"    {d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[9. ball1-ball2 동시 소수 패턴]")
    for pattern, freq in prime_pattern_freq.most_common():
        p1 = "소수" if pattern[0] == 'P' else "비소수"
        p2 = "소수" if pattern[1] == 'P' else "비소수"
        print(f"    {pattern}: ball1={p1:3s}, ball2={p2:3s} → {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[10. 최근 트렌드 (상승 번호)]")
    trending_up = []
    for v in ball2_freq.keys():
        all_ratio = ball2_freq[v] / total_rounds
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
