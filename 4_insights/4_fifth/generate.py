"""
Fifth(5번째 번호) 인사이트 생성 - 전문가 분석

★ 가장 중요한 분석 - ball5는 ball6(끝수) 직전 번호로 조합 결정에 핵심

핵심 분석 요소:
1. Sum(합계) 활용 - ball1~4 합계로 ball5 범위 예측
2. 소수 패턴 - ball1~4 소수 개수와 ball5 소수 여부 상관관계
3. 연속수 패턴 - ball4-ball5, ball5-ball6 연속 여부
4. Shortcode 활용 - Top24/Mid14/Rest7 세그먼트 패턴

분석 항목:
1. ball5 기본 빈도 분석
2. ball5 구간별 분포 (0-4)
3. ball1~4 합계별 ball5 분석 (핵심!)
4. ball1~4 소수 개수별 ball5 분석
5. ball4-ball5 연속수 분석
6. ball5-ball6 연속수 분석
7. ball5 자신의 소수 여부
8. ball5 홀짝 분석
9. ball6(끝수)과의 간격 분석
10. Shortcode 패턴별 ball5 분석
11. 복합 조건 분석 (sum + 소수 + 연속)
12. 최근 트렌드 분석
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import statistics
import sys

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
BACKTEST_PATH = Path(__file__).parent.parent.parent / "3_backtest" / "backtest_results.csv"
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
                'sum_1to4': balls[0] + balls[1] + balls[2] + balls[3],
                'sum_1to5': balls[0] + balls[1] + balls[2] + balls[3] + balls[4],
                'sum_total': sum(balls),
            })
    return results


def load_backtest_results() -> dict:
    """백테스트 결과 로드 (shortcode용)"""
    results = {}
    if not BACKTEST_PATH.exists():
        return results
    with open(BACKTEST_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[int(row['round'])] = row['summary']
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


def get_sum_range(s: int) -> str:
    """합계 구간 분류"""
    if s <= 40: return '~40'
    elif s <= 50: return '41-50'
    elif s <= 60: return '51-60'
    elif s <= 70: return '61-70'
    elif s <= 80: return '71-80'
    elif s <= 90: return '81-90'
    else: return '91~'


def generate_insight():
    """Fifth 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    shortcodes = load_backtest_results()
    total_rounds = len(data)

    # ========== 1. ball5 기본 빈도 분석 ==========
    ball5_freq = Counter(r['ball5'] for r in data)

    with open(OUTPUT_DIR / "ball5_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball5', 'frequency', 'ratio', 'probability', 'is_prime', 'rank'])
        sorted_items = sorted(ball5_freq.items(), key=lambda x: -x[1])
        for rank, (v, freq) in enumerate(sorted_items, 1):
            writer.writerow([v, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), is_prime(v), rank])

    # ========== 2. ball5 구간별 분포 ==========
    range_freq = Counter(get_range(r['ball5']) for r in data)

    with open(OUTPUT_DIR / "ball5_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'label', 'frequency', 'ratio', 'probability'])
        for rng in range(5):
            freq = range_freq.get(rng, 0)
            writer.writerow([rng, get_range_label(rng), freq,
                           round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 3. ball1~4 합계별 ball5 분석 (핵심!) ==========
    sum14_ball5 = defaultdict(list)
    for r in data:
        sum_range = get_sum_range(r['sum_1to4'])
        sum14_ball5[sum_range].append(r['ball5'])

    with open(OUTPUT_DIR / "sum14_ball5_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sum_1to4_range', 'count', 'ball5_avg', 'ball5_min', 'ball5_max',
                        'ball5_std', 'ball5_prime_ratio', 'recommended_ball5_range'])
        for sr in ['~40', '41-50', '51-60', '61-70', '71-80', '81-90', '91~']:
            balls = sum14_ball5.get(sr, [])
            if balls:
                avg = statistics.mean(balls)
                std = statistics.stdev(balls) if len(balls) > 1 else 0
                prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
                rec_min = max(int(avg - std), min(balls))
                rec_max = min(int(avg + std), max(balls))
                writer.writerow([sr, len(balls), round(avg, 1), min(balls), max(balls),
                               round(std, 2), round(prime_ratio, 1), f"{rec_min}-{rec_max}"])

    # ========== 4. ball1~4 소수 개수별 ball5 분석 ==========
    prime_count_ball5 = defaultdict(list)
    for r in data:
        pc = count_primes([r['ball1'], r['ball2'], r['ball3'], r['ball4']])
        prime_count_ball5[pc].append(r['ball5'])

    with open(OUTPUT_DIR / "ball1234_prime_count_ball5.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prime_count_in_1234', 'count', 'ball5_avg', 'ball5_min', 'ball5_max',
                        'ball5_std', 'ball5_prime_ratio'])
        for pc in sorted(prime_count_ball5.keys()):
            balls = prime_count_ball5[pc]
            prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
            writer.writerow([pc, len(balls), round(statistics.mean(balls), 1),
                           min(balls), max(balls),
                           round(statistics.stdev(balls), 2) if len(balls) > 1 else 0,
                           round(prime_ratio, 1)])

    # ========== 5. ball4-ball5 연속수 분석 ==========
    gap45_freq = Counter()
    consec45_freq = Counter()
    for r in data:
        gap = r['ball5'] - r['ball4']
        gap45_freq[gap] += 1
        consec45_freq[gap == 1] += 1

    with open(OUTPUT_DIR / "ball4_ball5_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'probability', 'is_consecutive'])
        for gap in sorted(gap45_freq.keys()):
            freq = gap45_freq[gap]
            writer.writerow([gap, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), gap == 1])

    # ========== 6. ball5-ball6 연속수 분석 ==========
    gap56_freq = Counter()
    consec56_freq = Counter()
    for r in data:
        gap = r['ball6'] - r['ball5']
        gap56_freq[gap] += 1
        consec56_freq[gap == 1] += 1

    with open(OUTPUT_DIR / "ball5_ball6_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'probability', 'is_consecutive'])
        for gap in sorted(gap56_freq.keys()):
            freq = gap56_freq[gap]
            writer.writerow([gap, freq, round(freq/total_rounds*100, 1),
                           round(freq/total_rounds, 4), gap == 1])

    # ========== 7. ball5 자신의 소수 여부 ==========
    ball5_prime_freq = Counter(is_prime(r['ball5']) for r in data)

    with open(OUTPUT_DIR / "ball5_prime_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['is_prime', 'frequency', 'ratio', 'probability'])
        for is_p in [True, False]:
            freq = ball5_prime_freq.get(is_p, 0)
            writer.writerow([is_p, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 8. ball5 홀짝 분석 ==========
    oddeven_freq = Counter('홀' if r['ball5'] % 2 == 1 else '짝' for r in data)

    with open(OUTPUT_DIR / "ball5_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'frequency', 'ratio', 'probability'])
        for t in ['홀', '짝']:
            freq = oddeven_freq.get(t, 0)
            writer.writerow([t, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 9. ball6(끝수)과의 간격 분석 ==========
    gap_to_end = defaultdict(list)
    for r in data:
        gap = r['ball6'] - r['ball5']
        gap_to_end[gap].append(r['ball5'])

    with open(OUTPUT_DIR / "ball5_gap_to_ball6.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap_to_ball6', 'count', 'ball5_avg', 'ball5_min', 'ball5_max', 'ratio'])
        for gap in sorted(gap_to_end.keys()):
            balls = gap_to_end[gap]
            writer.writerow([gap, len(balls), round(statistics.mean(balls), 1),
                           min(balls), max(balls), round(len(balls)/total_rounds*100, 1)])

    # ========== 10. Shortcode 패턴별 ball5 분석 ==========
    if shortcodes:
        # ord_code(앞 3자리)별 ball5 분석
        ord_code_ball5 = defaultdict(list)
        for r in data:
            if r['round'] in shortcodes:
                code = shortcodes[r['round']][:3]  # ord_code
                ord_code_ball5[code].append(r['ball5'])

        with open(OUTPUT_DIR / "shortcode_ord_ball5.csv", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ord_code', 'count', 'ball5_avg', 'ball5_min', 'ball5_max',
                            'ball5_prime_ratio', 'ratio'])
            sorted_codes = sorted(ord_code_ball5.items(), key=lambda x: -len(x[1]))
            for code, balls in sorted_codes[:20]:  # 상위 20개
                prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
                writer.writerow([code, len(balls), round(statistics.mean(balls), 1),
                               min(balls), max(balls), round(prime_ratio, 1),
                               round(len(balls)/total_rounds*100, 1)])

        # ball_code(뒤 3자리)별 ball5 분석
        ball_code_ball5 = defaultdict(list)
        for r in data:
            if r['round'] in shortcodes:
                code = shortcodes[r['round']][3:]  # ball_code
                ball_code_ball5[code].append(r['ball5'])

        with open(OUTPUT_DIR / "shortcode_ball_ball5.csv", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ball_code', 'count', 'ball5_avg', 'ball5_min', 'ball5_max',
                            'ball5_prime_ratio', 'ratio'])
            sorted_codes = sorted(ball_code_ball5.items(), key=lambda x: -len(x[1]))
            for code, balls in sorted_codes[:20]:
                prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
                writer.writerow([code, len(balls), round(statistics.mean(balls), 1),
                               min(balls), max(balls), round(prime_ratio, 1),
                               round(len(balls)/total_rounds*100, 1)])

    # ========== 11. 복합 조건 분석 (sum + 소수 + 연속) ==========
    complex_analysis = defaultdict(list)
    for r in data:
        sum_range = get_sum_range(r['sum_1to4'])
        prime_count = count_primes([r['ball1'], r['ball2'], r['ball3'], r['ball4']])
        has_consec_45 = r['ball5'] - r['ball4'] == 1
        has_consec_56 = r['ball6'] - r['ball5'] == 1
        key = (sum_range, prime_count, has_consec_45, has_consec_56)
        complex_analysis[key].append(r['ball5'])

    with open(OUTPUT_DIR / "complex_pattern_ball5.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sum_1to4_range', 'prime_count_1234', 'consec_45', 'consec_56',
                        'count', 'ball5_avg', 'ball5_std', 'ball5_prime_ratio', 'ratio'])
        sorted_patterns = sorted(complex_analysis.items(), key=lambda x: -len(x[1]))
        for (sr, pc, c45, c56), balls in sorted_patterns[:30]:
            if len(balls) >= 3:
                prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
                writer.writerow([sr, pc, c45, c56, len(balls),
                               round(statistics.mean(balls), 1),
                               round(statistics.stdev(balls), 2) if len(balls) > 1 else 0,
                               round(prime_ratio, 1), round(len(balls)/total_rounds*100, 1)])

    # ========== 12. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_freq = Counter(r['ball5'] for r in recent_50)
    recent_100_freq = Counter(r['ball5'] for r in recent_100)

    with open(OUTPUT_DIR / "ball5_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball5', 'is_prime', 'all_time_freq', 'all_time_ratio',
                        'recent_100_freq', 'recent_100_ratio',
                        'recent_50_freq', 'recent_50_ratio', 'trend'])
        for v in sorted(ball5_freq.keys()):
            all_freq = ball5_freq[v]
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
    all_ball5 = [r['ball5'] for r in data]
    median_ball5 = statistics.median(all_ball5)

    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['ball5_avg', round(statistics.mean(all_ball5), 2), 'ball5 평균'])
        writer.writerow(['ball5_std', round(statistics.stdev(all_ball5), 2), 'ball5 표준편차'])
        writer.writerow(['ball5_median', median_ball5, 'ball5 중앙값'])
        writer.writerow(['ball5_mode', ball5_freq.most_common(1)[0][0], 'ball5 최빈값'])
        writer.writerow(['ball5_min', min(all_ball5), 'ball5 최소값'])
        writer.writerow(['ball5_max', max(all_ball5), 'ball5 최대값'])
        writer.writerow(['ball5_prime_ratio', round(ball5_prime_freq[True]/total_rounds*100, 1),
                        'ball5 소수 비율(%)'])
        writer.writerow(['consecutive_45_ratio', round(consec45_freq[True]/total_rounds*100, 1),
                        'ball4-ball5 연속수 비율(%)'])
        writer.writerow(['consecutive_56_ratio', round(consec56_freq[True]/total_rounds*100, 1),
                        'ball5-ball6 연속수 비율(%)'])
        writer.writerow(['odd_ratio', round(oddeven_freq['홀']/total_rounds*100, 1),
                        'ball5 홀수 비율(%)'])

    # ========== 출력 ==========
    print("=" * 70)
    print("Fifth(5번째 번호) 전문가 인사이트")
    print("=" * 70)
    print(f"\n★ ball5는 ball6 직전 번호로 조합 결정의 핵심!")
    print(f"총 {total_rounds}회차 분석\n")

    print("[1. ball5 기본 통계]")
    print(f"    평균: {statistics.mean(all_ball5):.1f}")
    print(f"    표준편차: {statistics.stdev(all_ball5):.1f}")
    print(f"    중앙값: {median_ball5}")
    print(f"    최빈값: {ball5_freq.most_common(1)[0][0]} ({ball5_freq.most_common(1)[0][1]}회)")

    print("\n[2. ball5 빈도 상위 10개]")
    for v, c in ball5_freq.most_common(10):
        prime_mark = "★" if is_prime(v) else ""
        print(f"    {v:2d}{prime_mark}: {c:3d}회 ({c/total_rounds*100:5.1f}%)")

    print("\n[3. ball5 구간별 분포]")
    for rng in range(5):
        freq = range_freq.get(rng, 0)
        bar = '█' * int(freq/total_rounds*50)
        print(f"    {rng}({get_range_label(rng)}): {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[4. ★ ball1~4 합계별 ball5 (핵심!)]")
    for sr in ['~40', '41-50', '51-60', '61-70', '71-80', '81-90', '91~']:
        balls = sum14_ball5.get(sr, [])
        if balls:
            avg = statistics.mean(balls)
            std = statistics.stdev(balls) if len(balls) > 1 else 0
            rec_range = f"{max(int(avg-std), min(balls))}-{min(int(avg+std), max(balls))}"
            print(f"    합계 {sr:>5}: {len(balls):3d}회, ball5 평균={avg:5.1f}, 권장범위={rec_range}")

    print("\n[5. ball1~4 소수 개수별 ball5]")
    for pc in sorted(prime_count_ball5.keys()):
        balls = prime_count_ball5[pc]
        avg = statistics.mean(balls)
        prime_ratio = sum(1 for b in balls if is_prime(b)) / len(balls) * 100
        print(f"    소수 {pc}개: {len(balls):3d}회, ball5 평균={avg:.1f}, ball5 소수비율={prime_ratio:.1f}%")

    print("\n[6. 연속수 패턴]")
    print(f"    ball4-ball5 연속: {consec45_freq[True]:3d}회 ({consec45_freq[True]/total_rounds*100:.1f}%)")
    print(f"    ball5-ball6 연속: {consec56_freq[True]:3d}회 ({consec56_freq[True]/total_rounds*100:.1f}%)")

    print("\n[7. ball5 소수 여부]")
    for is_p in [True, False]:
        freq = ball5_prime_freq.get(is_p, 0)
        label = "소수" if is_p else "비소수"
        print(f"    {label}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[8. ball5 홀짝 분포]")
    for t in ['홀', '짝']:
        freq = oddeven_freq.get(t, 0)
        print(f"    {t}수: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[9. ball5-ball6 간격 분포 (상위 5개)]")
    for gap, freq in sorted(gap56_freq.items(), key=lambda x: -x[1])[:5]:
        label = "(연속)" if gap == 1 else ""
        print(f"    간격 {gap:2d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {label}")

    if shortcodes:
        print("\n[10. Shortcode ord_code별 ball5 (상위 5개)]")
        sorted_codes = sorted(ord_code_ball5.items(), key=lambda x: -len(x[1]))[:5]
        for code, balls in sorted_codes:
            avg = statistics.mean(balls)
            print(f"    {code}: {len(balls):3d}회, ball5 평균={avg:.1f}")

    print("\n[11. 최근 트렌드 (상승 번호)]")
    trending_up = []
    for v in ball5_freq.keys():
        all_ratio = ball5_freq[v] / total_rounds
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
