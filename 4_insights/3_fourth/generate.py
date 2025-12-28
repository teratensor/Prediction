"""
Fourth(4번째 번호) 인사이트 생성 - 전문가 분석

분석 항목:
1. ball4 기본 빈도 분석
2. ball4 구간별 분포 (0-4)
3. ball1,2,3 소수 개수별 ball4 분석
4. ball6(끝수) 소수 여부별 ball4 분석
5. ball3-ball4 연속수 분석
6. ball4 자신의 소수 여부 분석
7. ball4 홀짝 분석
8. ball4 끝자리(0-9) 분석
9. ball4 "적당한 수" 분석 (중앙값 근처)
10. ball1,2,3,6 복합 패턴과 ball4 관계
11. 최근 트렌드 분석
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


def get_range(n: int) -> int:
    """구간 반환 (0-4)"""
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def get_range_label(r: int) -> str:
    """구간 라벨"""
    labels = {0: '01-09', 1: '10-19', 2: '20-29', 3: '30-39', 4: '40-45'}
    return labels.get(r, str(r))


def is_prime(n: int) -> bool:
    return n in PRIMES


def count_primes(numbers: list) -> int:
    return sum(1 for n in numbers if is_prime(n))


def generate_insight():
    """Fourth 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # ========== 1. ball4 기본 빈도 분석 ==========
    ball4_freq = Counter(r['ball4'] for r in data)

    with open(OUTPUT_DIR / "ball4_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball4', 'frequency', 'ratio', 'probability', 'is_prime', 'rank'])
        sorted_items = sorted(ball4_freq.items(), key=lambda x: -x[1])
        for rank, (v, freq) in enumerate(sorted_items, 1):
            writer.writerow([v, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), is_prime(v), rank])

    # ========== 2. ball4 구간별 분포 ==========
    range_freq = Counter(get_range(r['ball4']) for r in data)

    with open(OUTPUT_DIR / "ball4_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'label', 'frequency', 'ratio', 'probability'])
        for rng in range(5):
            freq = range_freq.get(rng, 0)
            writer.writerow([rng, get_range_label(rng), freq,
                           round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 3. ball1,2,3 소수 개수별 ball4 분석 ==========
    prime_count_ball4 = defaultdict(list)
    for r in data:
        pc = count_primes([r['ball1'], r['ball2'], r['ball3']])
        prime_count_ball4[pc].append(r['ball4'])

    with open(OUTPUT_DIR / "ball123_prime_count_ball4.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prime_count_in_123', 'count', 'ball4_avg', 'ball4_min', 'ball4_max',
                        'ball4_std', 'ball4_prime_ratio'])
        for pc in sorted(prime_count_ball4.keys()):
            balls = prime_count_ball4[pc]
            prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
            writer.writerow([pc, len(balls), round(statistics.mean(balls), 1),
                           min(balls), max(balls),
                           round(statistics.stdev(balls), 2) if len(balls) > 1 else 0,
                           round(prime_ratio, 1)])

    # ========== 4. ball6(끝수) 소수 여부별 ball4 분석 ==========
    ball6_prime_ball4 = defaultdict(list)
    for r in data:
        key = 'P' if is_prime(r['ball6']) else 'N'
        ball6_prime_ball4[key].append(r['ball4'])

    with open(OUTPUT_DIR / "ball6_prime_ball4.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball6_prime', 'count', 'ball4_avg', 'ball4_min', 'ball4_max',
                        'ball4_std', 'ball4_prime_ratio'])
        for key in ['P', 'N']:
            balls = ball6_prime_ball4[key]
            if balls:
                prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
                writer.writerow([key == 'P', len(balls), round(statistics.mean(balls), 1),
                               min(balls), max(balls),
                               round(statistics.stdev(balls), 2) if len(balls) > 1 else 0,
                               round(prime_ratio, 1)])

    # ========== 5. ball3-ball4 연속수 분석 ==========
    gap34_freq = Counter()
    consecutive_freq = Counter()
    for r in data:
        gap = r['ball4'] - r['ball3']
        gap34_freq[gap] += 1
        consecutive_freq[gap == 1] += 1

    with open(OUTPUT_DIR / "ball3_ball4_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'probability', 'is_consecutive'])
        for gap in sorted(gap34_freq.keys()):
            freq = gap34_freq[gap]
            writer.writerow([gap, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), gap == 1])

    # ========== 6. ball4 자신의 소수 여부 분석 ==========
    ball4_prime_freq = Counter(is_prime(r['ball4']) for r in data)

    with open(OUTPUT_DIR / "ball4_prime_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['is_prime', 'frequency', 'ratio', 'probability'])
        for is_p in [True, False]:
            freq = ball4_prime_freq.get(is_p, 0)
            writer.writerow([is_p, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 7. ball4 홀짝 분석 ==========
    oddeven_freq = Counter('홀' if r['ball4'] % 2 == 1 else '짝' for r in data)

    with open(OUTPUT_DIR / "ball4_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'frequency', 'ratio', 'probability'])
        for t in ['홀', '짝']:
            freq = oddeven_freq.get(t, 0)
            writer.writerow([t, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 8. ball4 끝자리 분석 ==========
    lastdigit_freq = Counter(r['ball4'] % 10 for r in data)

    with open(OUTPUT_DIR / "ball4_lastdigit_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['last_digit', 'frequency', 'ratio', 'probability'])
        for d in range(10):
            freq = lastdigit_freq.get(d, 0)
            writer.writerow([d, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 9. ball4 "적당한 수" 분석 (중앙값 근처 여부) ==========
    # 적당한 수 = 전체 중앙값 ± 10 범위
    all_ball4 = [r['ball4'] for r in data]
    median_ball4 = statistics.median(all_ball4)

    appropriate_freq = Counter()
    for r in data:
        is_appropriate = abs(r['ball4'] - median_ball4) <= 10
        appropriate_freq[is_appropriate] += 1

    with open(OUTPUT_DIR / "ball4_appropriate_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['is_appropriate', 'median', 'range', 'frequency', 'ratio', 'probability'])
        for is_app in [True, False]:
            freq = appropriate_freq.get(is_app, 0)
            range_str = f"{int(median_ball4-10)}-{int(median_ball4+10)}" if is_app else "outside"
            writer.writerow([is_app, int(median_ball4), range_str, freq,
                           round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 10. ball1,2,3,6 복합 패턴과 ball4 관계 ==========
    # 패턴: (ball123 소수개수, ball6 소수여부, ball3-4 연속여부)
    complex_pattern_ball4 = defaultdict(list)
    for r in data:
        pc123 = count_primes([r['ball1'], r['ball2'], r['ball3']])
        b6_prime = 'P' if is_prime(r['ball6']) else 'N'
        consec = 'C' if r['ball4'] - r['ball3'] == 1 else 'N'
        key = (pc123, b6_prime, consec)
        complex_pattern_ball4[key].append(r['ball4'])

    with open(OUTPUT_DIR / "complex_pattern_ball4.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prime_count_123', 'ball6_prime', 'is_consecutive',
                        'count', 'ball4_avg', 'ball4_prime_ratio', 'ratio'])
        sorted_patterns = sorted(complex_pattern_ball4.items(), key=lambda x: -len(x[1]))
        for (pc, b6p, consec), balls in sorted_patterns:
            if len(balls) >= 5:  # 최소 5회 이상
                prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
                writer.writerow([pc, b6p == 'P', consec == 'C', len(balls),
                               round(statistics.mean(balls), 1), round(prime_ratio, 1),
                               round(len(balls)/total_rounds*100, 1)])

    # ========== 11. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_freq = Counter(r['ball4'] for r in recent_50)
    recent_100_freq = Counter(r['ball4'] for r in recent_100)

    with open(OUTPUT_DIR / "ball4_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball4', 'is_prime', 'all_time_freq', 'all_time_ratio',
                        'recent_100_freq', 'recent_100_ratio',
                        'recent_50_freq', 'recent_50_ratio', 'trend'])
        for v in sorted(ball4_freq.keys()):
            all_freq = ball4_freq[v]
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

            writer.writerow([v, is_prime(v), all_freq, round(all_ratio*100, 1),
                           r100_freq, round(r100_ratio*100, 1),
                           r50_freq, round(r50_ratio*100, 1), trend])

    # ========== Summary ==========
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['ball4_avg', round(statistics.mean(all_ball4), 2), 'ball4 평균'])
        writer.writerow(['ball4_std', round(statistics.stdev(all_ball4), 2), 'ball4 표준편차'])
        writer.writerow(['ball4_median', median_ball4, 'ball4 중앙값'])
        writer.writerow(['ball4_mode', ball4_freq.most_common(1)[0][0], 'ball4 최빈값'])
        writer.writerow(['ball4_min', min(all_ball4), 'ball4 최소값'])
        writer.writerow(['ball4_max', max(all_ball4), 'ball4 최대값'])
        writer.writerow(['ball4_prime_ratio', round(ball4_prime_freq[True]/total_rounds*100, 1),
                        'ball4 소수 비율(%)'])
        writer.writerow(['consecutive_34_ratio', round(consecutive_freq[True]/total_rounds*100, 1),
                        'ball3-ball4 연속수 비율(%)'])
        writer.writerow(['odd_ratio', round(oddeven_freq['홀']/total_rounds*100, 1),
                        'ball4 홀수 비율(%)'])
        writer.writerow(['appropriate_ratio', round(appropriate_freq[True]/total_rounds*100, 1),
                        f'ball4 적당한수({int(median_ball4-10)}-{int(median_ball4+10)}) 비율(%)'])

    # ========== 출력 ==========
    print("=" * 70)
    print("Fourth(4번째 번호) 전문가 인사이트")
    print("=" * 70)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. ball4 기본 통계]")
    print(f"    평균: {statistics.mean(all_ball4):.1f}")
    print(f"    표준편차: {statistics.stdev(all_ball4):.1f}")
    print(f"    중앙값: {median_ball4}")
    print(f"    최빈값: {ball4_freq.most_common(1)[0][0]} ({ball4_freq.most_common(1)[0][1]}회)")

    print("\n[2. ball4 빈도 상위 10개]")
    for v, c in ball4_freq.most_common(10):
        prime_mark = "★" if is_prime(v) else ""
        print(f"    {v:2d}{prime_mark}: {c:3d}회 ({c/total_rounds*100:5.1f}%)")

    print("\n[3. ball4 구간별 분포]")
    for rng in range(5):
        freq = range_freq.get(rng, 0)
        bar = '█' * int(freq/total_rounds*50)
        print(f"    {rng}({get_range_label(rng)}): {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[4. ball1,2,3 소수 개수별 ball4]")
    for pc in sorted(prime_count_ball4.keys()):
        balls = prime_count_ball4[pc]
        avg = statistics.mean(balls)
        prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
        print(f"    소수 {pc}개: {len(balls):3d}회, ball4 평균={avg:.1f}, ball4 소수비율={prime_ratio:.1f}%")

    print("\n[5. ball6(끝수) 소수 여부별 ball4]")
    for key in ['P', 'N']:
        balls = ball6_prime_ball4[key]
        if balls:
            label = "소수" if key == 'P' else "비소수"
            avg = statistics.mean(balls)
            prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
            print(f"    ball6 {label}: {len(balls):3d}회, ball4 평균={avg:.1f}, ball4 소수비율={prime_ratio:.1f}%")

    print("\n[6. ball3-ball4 간격 분포 (상위 5개)]")
    for gap, freq in sorted(gap34_freq.items(), key=lambda x: -x[1])[:5]:
        label = "(연속)" if gap == 1 else ""
        print(f"    간격 {gap:2d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {label}")

    print("\n[7. ball4 소수 여부]")
    for is_p in [True, False]:
        freq = ball4_prime_freq.get(is_p, 0)
        label = "소수" if is_p else "비소수"
        print(f"    {label}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[8. ball4 홀짝 분포]")
    for t in ['홀', '짝']:
        freq = oddeven_freq.get(t, 0)
        print(f"    {t}수: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print(f"\n[9. ball4 적당한 수 (중앙값 {int(median_ball4)} ± 10)]")
    for is_app in [True, False]:
        freq = appropriate_freq.get(is_app, 0)
        label = f"{int(median_ball4-10)}-{int(median_ball4+10)}" if is_app else "범위 외"
        print(f"    {label}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[10. 최근 트렌드 (상승 번호)]")
    trending_up = []
    for v in ball4_freq.keys():
        all_ratio = ball4_freq[v] / total_rounds
        r50_ratio = recent_50_freq.get(v, 0) / 50
        if r50_ratio > all_ratio * 1.3:
            trending_up.append((v, r50_ratio * 100, all_ratio * 100, is_prime(v)))
    for v, r50, all_r, is_p in sorted(trending_up, key=lambda x: -x[1])[:5]:
        prime_mark = "★" if is_p else ""
        print(f"    {v:2d}{prime_mark}: 최근50회 {r50:.1f}% (전체 {all_r:.1f}%) ↑")

    print("\n" + "=" * 70)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_insight()
