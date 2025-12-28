"""
Range(구간) 인사이트 생성 - 전문가 분석

구간 정의:
- 0: 1-9 (9개)
- 1: 10-19 (10개)
- 2: 20-29 (10개)
- 3: 30-39 (10개)
- 4: 40-45 (6개)

분석 항목:
1. 구간별 기본 빈도 분석
2. 6자리 구간 코드 패턴 분석 (예: 011234)
3. 구간별 번호 출현 빈도
4. 구간 조합 패턴 (몇 개 구간 사용?)
5. 연속 구간 패턴 분석
6. 구간별 홀짝 분포
7. 구간 점프 패턴 (건너뛰는 구간)
8. 최근 트렌드 분석
9. 구간 코드 → 이론적 조합 수 vs 실제 빈도
10. 포지션별 구간 분포 (ball1~ball6)
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
import statistics
import math

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"

# 구간별 번호 개수
RANGE_SIZE = {0: 9, 1: 10, 2: 10, 3: 10, 4: 6}  # 총 45개


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
    """번호 → 구간 변환"""
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def get_range_label(r: int) -> str:
    """구간 → 라벨"""
    labels = {0: '01-09', 1: '10-19', 2: '20-29', 3: '30-39', 4: '40-45'}
    return labels[r]


def get_range_code(balls: list) -> str:
    """6개 번호 → 구간 코드 (예: 011234)"""
    return ''.join(str(get_range(b)) for b in balls)


def get_range_pattern(balls: list) -> str:
    """구간 분포 패턴 (예: 1-2-1-1-1 = 각 구간별 개수)"""
    ranges = [get_range(b) for b in balls]
    counts = [ranges.count(i) for i in range(5)]
    return '-'.join(map(str, counts))


def calc_theoretical_combinations(range_code: str) -> int:
    """구간 코드의 이론적 조합 수 계산"""
    # 구간별 사용 횟수 계산
    range_counts = Counter(range_code)

    # 각 구간에서 선택하는 조합 수
    total = 1
    for r, count in range_counts.items():
        r_int = int(r)
        n = RANGE_SIZE[r_int]
        total *= math.comb(n, count)

    return total


def generate_insight():
    """Range 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)

    # ========== 1. 구간별 기본 빈도 분석 ==========
    all_ranges = []
    for r in data:
        for b in r['balls']:
            all_ranges.append(get_range(b))

    range_freq = Counter(all_ranges)
    total_balls = len(all_ranges)

    # 이론적 확률 (각 구간 번호 수 / 45)
    theoretical_prob = {r: RANGE_SIZE[r] / 45 for r in range(5)}

    with open(OUTPUT_DIR / "range_frequency.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'label', 'numbers_in_range', 'frequency', 'ratio',
                        'theoretical_prob', 'actual_prob', 'deviation'])
        for r in range(5):
            freq = range_freq.get(r, 0)
            actual_prob = freq / total_balls
            deviation = (actual_prob - theoretical_prob[r]) / theoretical_prob[r] * 100
            writer.writerow([r, get_range_label(r), RANGE_SIZE[r], freq,
                           round(freq/total_balls*100, 2),
                           round(theoretical_prob[r]*100, 2),
                           round(actual_prob*100, 2),
                           round(deviation, 2)])

    # ========== 2. 6자리 구간 코드 패턴 ==========
    code_freq = Counter(get_range_code(r['balls']) for r in data)

    with open(OUTPUT_DIR / "range_code_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range_code', 'frequency', 'ratio', 'probability',
                        'theoretical_combinations', 'rank'])
        sorted_codes = sorted(code_freq.items(), key=lambda x: -x[1])
        for rank, (code, freq) in enumerate(sorted_codes, 1):
            theo_comb = calc_theoretical_combinations(code)
            writer.writerow([code, freq, round(freq/total_rounds*100, 2),
                           round(freq/total_rounds, 4), theo_comb, rank])

    # ========== 3. 구간 분포 패턴 (각 구간별 개수) ==========
    pattern_freq = Counter(get_range_pattern(r['balls']) for r in data)

    with open(OUTPUT_DIR / "range_pattern_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pattern', 'r0_count', 'r1_count', 'r2_count', 'r3_count', 'r4_count',
                        'frequency', 'ratio', 'probability'])
        for pattern, freq in pattern_freq.most_common():
            counts = pattern.split('-')
            writer.writerow([pattern] + counts + [freq, round(freq/total_rounds*100, 2),
                           round(freq/total_rounds, 4)])

    # ========== 4. 사용 구간 수 분석 ==========
    unique_ranges_freq = Counter()
    for r in data:
        ranges = set(get_range(b) for b in r['balls'])
        unique_ranges_freq[len(ranges)] += 1

    with open(OUTPUT_DIR / "unique_ranges_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['unique_ranges', 'frequency', 'ratio', 'probability'])
        for n in sorted(unique_ranges_freq.keys()):
            freq = unique_ranges_freq[n]
            writer.writerow([n, freq, round(freq/total_rounds*100, 2), round(freq/total_rounds, 4)])

    # ========== 5. 연속 구간 패턴 ==========
    consecutive_pattern_freq = Counter()
    for r in data:
        ranges = sorted(set(get_range(b) for b in r['balls']))
        # 연속 구간 체크
        is_all_consecutive = all(ranges[i+1] - ranges[i] == 1 for i in range(len(ranges)-1))
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(ranges)):
            if ranges[i] - ranges[i-1] == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        consecutive_pattern_freq[(len(ranges), max_consecutive, is_all_consecutive)] += 1

    with open(OUTPUT_DIR / "consecutive_ranges_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['unique_ranges', 'max_consecutive', 'all_consecutive',
                        'frequency', 'ratio', 'probability'])
        for (n, max_c, all_c), freq in sorted(consecutive_pattern_freq.items(), key=lambda x: -x[1]):
            writer.writerow([n, max_c, all_c, freq, round(freq/total_rounds*100, 2),
                           round(freq/total_rounds, 4)])

    # ========== 6. 포지션별 구간 분포 ==========
    position_range_freq = {i: Counter() for i in range(1, 7)}
    for r in data:
        for i in range(1, 7):
            position_range_freq[i][get_range(r[f'ball{i}'])] += 1

    with open(OUTPUT_DIR / "position_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'range', 'label', 'frequency', 'ratio', 'probability'])
        for pos in range(1, 7):
            for r in range(5):
                freq = position_range_freq[pos].get(r, 0)
                writer.writerow([pos, r, get_range_label(r), freq,
                               round(freq/total_rounds*100, 2), round(freq/total_rounds, 4)])

    # ========== 7. 구간 점프 패턴 (건너뛰는 구간) ==========
    skip_pattern_freq = Counter()
    for r in data:
        ranges = sorted(set(get_range(b) for b in r['balls']))
        skipped = set(range(min(ranges), max(ranges)+1)) - set(ranges)
        skip_pattern_freq[tuple(sorted(skipped))] += 1

    with open(OUTPUT_DIR / "range_skip_pattern.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skipped_ranges', 'skip_count', 'frequency', 'ratio', 'probability'])
        for skipped, freq in skip_pattern_freq.most_common():
            skip_str = ','.join(map(str, skipped)) if skipped else 'none'
            writer.writerow([skip_str, len(skipped), freq, round(freq/total_rounds*100, 2),
                           round(freq/total_rounds, 4)])

    # ========== 8. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    all_code_freq = code_freq
    recent_50_code_freq = Counter(get_range_code(r['balls']) for r in recent_50)
    recent_100_code_freq = Counter(get_range_code(r['balls']) for r in recent_100)

    with open(OUTPUT_DIR / "range_code_trend.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range_code', 'all_time_freq', 'all_time_ratio',
                        'recent_100_freq', 'recent_100_ratio',
                        'recent_50_freq', 'recent_50_ratio', 'trend'])

        # 상위 30개 코드에 대해
        for code, _ in all_code_freq.most_common(30):
            all_freq = all_code_freq[code]
            r100_freq = recent_100_code_freq.get(code, 0)
            r50_freq = recent_50_code_freq.get(code, 0)

            all_ratio = all_freq / total_rounds
            r100_ratio = r100_freq / 100 if r100_freq else 0
            r50_ratio = r50_freq / 50 if r50_freq else 0

            if r50_ratio > all_ratio * 1.5:
                trend = '급상승'
            elif r50_ratio > all_ratio * 1.2:
                trend = '상승'
            elif r50_ratio < all_ratio * 0.5:
                trend = '급하락'
            elif r50_ratio < all_ratio * 0.8:
                trend = '하락'
            else:
                trend = '보합'

            writer.writerow([code, all_freq, round(all_ratio*100, 2),
                           r100_freq, round(r100_ratio*100, 2),
                           r50_freq, round(r50_ratio*100, 2), trend])

    # ========== 9. 구간별 홀짝 분포 ==========
    range_oddeven = {r: {'홀': 0, '짝': 0} for r in range(5)}
    for r in data:
        for b in r['balls']:
            rng = get_range(b)
            oe = '홀' if b % 2 == 1 else '짝'
            range_oddeven[rng][oe] += 1

    with open(OUTPUT_DIR / "range_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'label', 'odd_count', 'even_count', 'odd_ratio', 'even_ratio'])
        for r in range(5):
            odd = range_oddeven[r]['홀']
            even = range_oddeven[r]['짝']
            total = odd + even
            writer.writerow([r, get_range_label(r), odd, even,
                           round(odd/total*100, 2) if total else 0,
                           round(even/total*100, 2) if total else 0])

    # ========== 10. 이론적 조합 수 vs 실제 빈도 TOP 20 ==========
    code_analysis = []
    for code, freq in code_freq.items():
        theo_comb = calc_theoretical_combinations(code)
        expected_ratio = theo_comb / 8145060  # 전체 조합 수
        actual_ratio = freq / total_rounds
        deviation = (actual_ratio - expected_ratio) / expected_ratio * 100 if expected_ratio > 0 else 0
        code_analysis.append({
            'code': code,
            'frequency': freq,
            'theoretical_combinations': theo_comb,
            'expected_ratio': expected_ratio,
            'actual_ratio': actual_ratio,
            'deviation': deviation
        })

    with open(OUTPUT_DIR / "range_code_theory_vs_actual.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range_code', 'frequency', 'theoretical_combinations',
                        'expected_ratio_pct', 'actual_ratio_pct', 'deviation_pct'])
        # 편차가 큰 순으로 정렬
        for item in sorted(code_analysis, key=lambda x: -abs(x['deviation']))[:30]:
            writer.writerow([item['code'], item['frequency'], item['theoretical_combinations'],
                           round(item['expected_ratio']*100, 4),
                           round(item['actual_ratio']*100, 2),
                           round(item['deviation'], 1)])

    # ========== Summary ==========
    top_code = code_freq.most_common(1)[0]
    avg_unique_ranges = statistics.mean(len(set(get_range(b) for b in r['balls'])) for r in data)

    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['unique_codes', len(code_freq), '출현한 구간코드 종류 수'])
        writer.writerow(['top_code', top_code[0], '최다 출현 구간코드'])
        writer.writerow(['top_code_freq', top_code[1], '최다 출현 구간코드 빈도'])
        writer.writerow(['avg_unique_ranges', round(avg_unique_ranges, 2), '평균 사용 구간 수'])
        for r in range(5):
            freq = range_freq.get(r, 0)
            writer.writerow([f'range_{r}_ratio', round(freq/total_balls*100, 2),
                           f'구간 {r} ({get_range_label(r)}) 출현 비율(%)'])

    # ========== 출력 ==========
    print("=" * 70)
    print("Range(구간) 전문가 인사이트")
    print("=" * 70)
    print(f"\n구간 정의: 0(1-9), 1(10-19), 2(20-29), 3(30-39), 4(40-45)")
    print(f"총 {total_rounds}회차 분석\n")

    print("[1. 구간별 출현 빈도]")
    print(f"    {'구간':<10} {'번호수':>6} {'빈도':>8} {'실제%':>8} {'이론%':>8} {'편차':>8}")
    print("    " + "-" * 50)
    for r in range(5):
        freq = range_freq.get(r, 0)
        actual = freq / total_balls * 100
        theo = theoretical_prob[r] * 100
        dev = (actual - theo) / theo * 100
        bar = '█' * int(freq/total_balls*50)
        print(f"    {r}({get_range_label(r):<5}) {RANGE_SIZE[r]:>6} {freq:>8} {actual:>7.1f}% {theo:>7.1f}% {dev:>+7.1f}%")

    print("\n[2. 구간코드 상위 15개]")
    print(f"    {'코드':<10} {'빈도':>6} {'비율':>8} {'이론조합':>12}")
    print("    " + "-" * 40)
    for code, freq in code_freq.most_common(15):
        theo_comb = calc_theoretical_combinations(code)
        print(f"    {code:<10} {freq:>6} {freq/total_rounds*100:>7.1f}% {theo_comb:>12,}")

    print("\n[3. 사용 구간 수 분포]")
    for n in sorted(unique_ranges_freq.keys()):
        freq = unique_ranges_freq[n]
        bar = '█' * int(freq/total_rounds*30)
        print(f"    {n}개 구간: {freq:>4}회 ({freq/total_rounds*100:>5.1f}%) {bar}")

    print("\n[4. 구간 분포 패턴 상위 10개]")
    print(f"    {'패턴':<15} {'의미':<30} {'빈도':>6} {'비율':>8}")
    print("    " + "-" * 65)
    for pattern, freq in pattern_freq.most_common(10):
        counts = [int(x) for x in pattern.split('-')]
        meaning = ', '.join([f"구간{i}:{c}개" for i, c in enumerate(counts) if c > 0])
        print(f"    {pattern:<15} {meaning:<30} {freq:>6} {freq/total_rounds*100:>7.1f}%")

    print("\n[5. 포지션별 주요 구간]")
    for pos in range(1, 7):
        top_range = position_range_freq[pos].most_common(1)[0]
        top2_range = position_range_freq[pos].most_common(2)[1] if len(position_range_freq[pos]) > 1 else (0, 0)
        print(f"    ball{pos}: 구간{top_range[0]}({get_range_label(top_range[0])}) {top_range[1]/total_rounds*100:.1f}%, "
              f"구간{top2_range[0]}({get_range_label(top2_range[0])}) {top2_range[1]/total_rounds*100:.1f}%")

    print("\n[6. 건너뛴 구간 패턴 상위 5개]")
    for skipped, freq in skip_pattern_freq.most_common(5):
        skip_str = ','.join([f"구간{s}" for s in skipped]) if skipped else '없음(연속)'
        print(f"    {skip_str:<20}: {freq:>4}회 ({freq/total_rounds*100:>5.1f}%)")

    print("\n[7. 최근 트렌드 (상승 코드)]")
    trending = []
    for code, _ in all_code_freq.most_common(50):
        all_ratio = all_code_freq[code] / total_rounds
        r50_ratio = recent_50_code_freq.get(code, 0) / 50
        if r50_ratio > all_ratio * 1.2:
            trending.append((code, r50_ratio*100, all_ratio*100))
    for code, r50, all_r in sorted(trending, key=lambda x: -x[1])[:5]:
        print(f"    {code}: 최근50회 {r50:.1f}% (전체 {all_r:.1f}%) ↑")

    print("\n" + "=" * 70)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_insight()
