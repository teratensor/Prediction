"""
Third(3번째 번호) 인사이트 생성 - 전문가 분석

분석 항목:
1. ball3 기본 빈도 분석
2. ball3 구간별 분포 (1-9, 10-19, 20-29, 30-39, 40-45)
3. ball2-ball3 연속수 분석
4. ball1, ball2 소수 패턴 분석
5. ball1-ball2 간격과 ball3 관계 분석
6. ball3의 홀짝 분석
7. ball3 끝자리(0-9) 분석
8. (ball1, ball2, ball3) 삼중 패턴 분석
9. 최근 N회 트렌드 분석
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
    """구간 라벨"""
    if n <= 9: return '01-09'
    elif n <= 19: return '10-19'
    elif n <= 29: return '20-29'
    elif n <= 39: return '30-39'
    else: return '40-45'


def is_consecutive(a: int, b: int) -> bool:
    return abs(a - b) == 1


def is_prime(n: int) -> bool:
    return n in PRIMES


def generate_insight():
    """Third 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # ========== 1. ball3 기본 빈도 분석 ==========
    ball3_freq = Counter(r['ball3'] for r in data)

    with open(OUTPUT_DIR / "ball3_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball3', 'frequency', 'ratio', 'probability', 'rank'])
        sorted_items = sorted(ball3_freq.items(), key=lambda x: -x[1])
        for rank, (v, freq) in enumerate(sorted_items, 1):
            writer.writerow([v, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4), rank])

    # ========== 2. ball3 구간별 분포 ==========
    range_freq = Counter(get_range_label(r['ball3']) for r in data)

    with open(OUTPUT_DIR / "ball3_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'frequency', 'ratio', 'probability'])
        for rng in ['01-09', '10-19', '20-29', '30-39', '40-45']:
            freq = range_freq.get(rng, 0)
            writer.writerow([rng, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 3. ball2-ball3 연속수 분석 ==========
    consecutive_freq = Counter()
    gap_freq = Counter()  # ball2-ball3 간격 분포
    for r in data:
        gap = r['ball3'] - r['ball2']
        gap_freq[gap] += 1
        consecutive_freq[gap == 1] += 1

    with open(OUTPUT_DIR / "ball2_ball3_gap_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['gap', 'frequency', 'ratio', 'probability', 'is_consecutive'])
        for gap in sorted(gap_freq.keys()):
            freq = gap_freq[gap]
            writer.writerow([gap, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4), gap == 1])

    # ========== 4. ball1, ball2 소수 패턴 ==========
    prime_pattern_freq = Counter()
    for r in data:
        p1 = 'P' if is_prime(r['ball1']) else 'N'
        p2 = 'P' if is_prime(r['ball2']) else 'N'
        prime_pattern_freq[p1 + p2] += 1

    with open(OUTPUT_DIR / "prime_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern', 'ball1_prime', 'ball2_prime', 'frequency', 'ratio', 'probability'])
        for pattern in ['PP', 'PN', 'NP', 'NN']:
            freq = prime_pattern_freq.get(pattern, 0)
            writer.writerow([pattern, pattern[0] == 'P', pattern[1] == 'P', freq,
                           round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 5. ball1-ball2 간격과 ball3 관계 ==========
    gap12_ball3 = defaultdict(list)
    for r in data:
        gap12 = r['ball2'] - r['ball1']
        gap12_ball3[gap12].append(r['ball3'])

    with open(OUTPUT_DIR / "gap12_ball3_relation.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball1_ball2_gap', 'count', 'ball3_avg', 'ball3_min', 'ball3_max', 'ball3_std'])
        for gap in sorted(gap12_ball3.keys()):
            balls = gap12_ball3[gap]
            if len(balls) >= 3:
                writer.writerow([gap, len(balls), round(statistics.mean(balls), 1),
                               min(balls), max(balls), round(statistics.stdev(balls), 2)])

    # ========== 6. ball3 홀짝 분석 ==========
    oddeven_freq = Counter('홀' if r['ball3'] % 2 == 1 else '짝' for r in data)

    with open(OUTPUT_DIR / "ball3_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'frequency', 'ratio', 'probability'])
        for t in ['홀', '짝']:
            freq = oddeven_freq.get(t, 0)
            writer.writerow([t, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 7. ball3 끝자리 분석 ==========
    lastdigit_freq = Counter(r['ball3'] % 10 for r in data)

    with open(OUTPUT_DIR / "ball3_lastdigit_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['last_digit', 'frequency', 'ratio', 'probability'])
        for d in range(10):
            freq = lastdigit_freq.get(d, 0)
            writer.writerow([d, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 8. (ball1, ball2) 조합별 ball3 분석 ==========
    pair12_ball3 = defaultdict(list)
    for r in data:
        pair12_ball3[(r['ball1'], r['ball2'])].append(r['ball3'])

    with open(OUTPUT_DIR / "pair12_ball3_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball1', 'ball2', 'count', 'ball3_values', 'ball3_avg', 'ball3_mode'])
        sorted_pairs = sorted(pair12_ball3.items(), key=lambda x: -len(x[1]))
        for (b1, b2), balls in sorted_pairs[:50]:  # 상위 50개
            mode = Counter(balls).most_common(1)[0][0]
            writer.writerow([b1, b2, len(balls), ','.join(map(str, sorted(balls))),
                           round(statistics.mean(balls), 1), mode])

    # ========== 9. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_freq = Counter(r['ball3'] for r in recent_50)
    recent_100_freq = Counter(r['ball3'] for r in recent_100)

    with open(OUTPUT_DIR / "ball3_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ball3', 'all_time_freq', 'all_time_ratio', 'recent_100_freq',
                        'recent_100_ratio', 'recent_50_freq', 'recent_50_ratio', 'trend'])
        for v in sorted(ball3_freq.keys()):
            all_freq = ball3_freq[v]
            r100_freq = recent_100_freq.get(v, 0)
            r50_freq = recent_50_freq.get(v, 0)

            all_ratio = all_freq / total_rounds
            r100_ratio = r100_freq / 100 if r100_freq else 0
            r50_ratio = r50_freq / 50 if r50_freq else 0

            # 트렌드 계산 (최근 50회 비율 vs 전체 비율)
            if r50_ratio > all_ratio * 1.3:
                trend = '상승'
            elif r50_ratio < all_ratio * 0.7:
                trend = '하락'
            else:
                trend = '보합'

            writer.writerow([v, all_freq, round(all_ratio*100, 1), r100_freq,
                           round(r100_ratio*100, 1), r50_freq, round(r50_ratio*100, 1), trend])

    # ========== 10. 복합 조건 분석 ==========
    complex_freq = Counter()
    for r in data:
        range_label = get_range_label(r['ball3'])
        is_consec = r['ball3'] - r['ball2'] == 1
        oddeven = '홀' if r['ball3'] % 2 == 1 else '짝'
        key = (range_label, is_consec, oddeven)
        complex_freq[key] += 1

    with open(OUTPUT_DIR / "complex_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'is_consecutive', 'oddeven', 'frequency', 'ratio', 'probability'])
        for (rng, consec, oe), freq in complex_freq.most_common():
            writer.writerow([rng, consec, oe, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== Summary ==========
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['ball3_avg', round(statistics.mean(r['ball3'] for r in data), 2), 'ball3 평균'])
        writer.writerow(['ball3_std', round(statistics.stdev(r['ball3'] for r in data), 2), 'ball3 표준편차'])
        writer.writerow(['ball3_median', statistics.median(r['ball3'] for r in data), 'ball3 중앙값'])
        writer.writerow(['ball3_mode', ball3_freq.most_common(1)[0][0], 'ball3 최빈값'])
        writer.writerow(['ball3_min', min(r['ball3'] for r in data), 'ball3 최소값'])
        writer.writerow(['ball3_max', max(r['ball3'] for r in data), 'ball3 최대값'])
        writer.writerow(['consecutive_ratio', round(consecutive_freq[True]/total_rounds*100, 1), 'ball2-ball3 연속수 비율(%)'])
        writer.writerow(['prime_NN_ratio', round(prime_pattern_freq['NN']/total_rounds*100, 1), 'ball1,ball2 둘다 비소수 비율(%)'])
        writer.writerow(['odd_ratio', round(oddeven_freq['홀']/total_rounds*100, 1), 'ball3 홀수 비율(%)'])
        writer.writerow(['range_10_19_ratio', round(range_freq['10-19']/total_rounds*100, 1), 'ball3 10-19 구간 비율(%)'])

    # ========== 출력 ==========
    print("=" * 60)
    print("Third(3번째 번호) 전문가 인사이트")
    print("=" * 60)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. ball3 기본 통계]")
    print(f"    평균: {statistics.mean(r['ball3'] for r in data):.1f}")
    print(f"    표준편차: {statistics.stdev(r['ball3'] for r in data):.1f}")
    print(f"    중앙값: {statistics.median(r['ball3'] for r in data)}")
    print(f"    최빈값: {ball3_freq.most_common(1)[0][0]} ({ball3_freq.most_common(1)[0][1]}회)")

    print("\n[2. ball3 빈도 상위 10개]")
    for v, c in ball3_freq.most_common(10):
        print(f"    {v:2d}: {c:3d}회 ({c/total_rounds*100:5.1f}%)")

    print("\n[3. ball3 구간별 분포]")
    for rng in ['01-09', '10-19', '20-29', '30-39', '40-45']:
        freq = range_freq.get(rng, 0)
        bar = '█' * int(freq/total_rounds*50)
        print(f"    {rng}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[4. ball2-ball3 간격 분포 (상위 5개)]")
    for gap, freq in sorted(gap_freq.items(), key=lambda x: -x[1])[:5]:
        label = "(연속)" if gap == 1 else ""
        print(f"    간격 {gap:2d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {label}")

    print("\n[5. ball1-ball2 소수 패턴]")
    for pattern, freq in prime_pattern_freq.most_common():
        p1 = "소수" if pattern[0] == 'P' else "비소수"
        p2 = "소수" if pattern[1] == 'P' else "비소수"
        print(f"    {pattern}: ball1={p1:3s}, ball2={p2:3s} → {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[6. ball3 홀짝 분포]")
    for t in ['홀', '짝']:
        freq = oddeven_freq.get(t, 0)
        print(f"    {t}수: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[7. ball3 끝자리 분포]")
    for d in range(10):
        freq = lastdigit_freq.get(d, 0)
        bar = '█' * int(freq/total_rounds*30)
        print(f"    {d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[8. 최근 트렌드 (상승 번호)]")
    trending_up = []
    for v in ball3_freq.keys():
        all_ratio = ball3_freq[v] / total_rounds
        r50_ratio = recent_50_freq.get(v, 0) / 50
        if r50_ratio > all_ratio * 1.3:
            trending_up.append((v, r50_ratio * 100, all_ratio * 100))
    for v, r50, all_r in sorted(trending_up, key=lambda x: -x[1])[:5]:
        print(f"    {v:2d}: 최근50회 {r50:.1f}% (전체 {all_r:.1f}%) ↑")

    print("\n" + "=" * 60)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_insight()
