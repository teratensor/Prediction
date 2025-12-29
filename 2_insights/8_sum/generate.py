"""
Sum(합계) 인사이트 생성 - 전문가 분석

분석 항목:
1. 합계 기본 통계 (평균, 표준편차, 중앙값, 사분위수)
2. 합계 빈도 분포
3. 합계 구간별 분포 (21-80, 81-120, 121-160, 161-200, 201-255)
4. 합계별 이론적 조합 수 vs 실제 출현 빈도 비교
5. 합계 끝자리(0-9) 분석
6. 홀수합/짝수합 분석
7. 최근 트렌드 분석 (전체 vs 최근 100회 vs 최근 50회)
8. 합계와 첫수/끝수 상관관계
9. 연속 회차 합계 변화량 분석
10. 합계 구간별 당첨 패턴
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import statistics
from itertools import combinations

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent / "statistics"


def load_data() -> list:
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            total = sum(balls)
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'ball1': balls[0],
                'ball6': balls[5],
                'sum': total
            })
    return results


def get_sum_range_label(s: int) -> str:
    """합계 구간 라벨"""
    if s <= 80: return '021-080'
    elif s <= 120: return '081-120'
    elif s <= 160: return '121-160'
    elif s <= 200: return '161-200'
    else: return '201-255'


def get_theoretical_combinations() -> dict:
    """이론적 합계별 조합 수 계산"""
    sum_counts = Counter()
    for combo in combinations(range(1, 46), 6):
        sum_counts[sum(combo)] += 1
    return sum_counts


def generate_insight():
    """Sum 인사이트 생성"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    total_rounds = len(data)
    sums = [r['sum'] for r in data]

    # 이론적 조합 수
    print("이론적 조합 수 계산 중...")
    theoretical = get_theoretical_combinations()
    total_combos = sum(theoretical.values())  # 8,145,060

    # ========== 1. 합계 기본 통계 ==========
    sum_mean = statistics.mean(sums)
    sum_std = statistics.stdev(sums)
    sum_median = statistics.median(sums)
    quartiles = statistics.quantiles(sums, n=4)

    # ========== 2. 합계 빈도 분포 ==========
    sum_freq = Counter(sums)

    with open(OUTPUT_DIR / "sum_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sum', 'frequency', 'ratio', 'probability', 'theoretical_combos', 'theoretical_ratio', 'observed_vs_expected'])
        for s in sorted(sum_freq.keys()):
            freq = sum_freq[s]
            theo = theoretical.get(s, 0)
            theo_ratio = theo / total_combos if theo else 0
            expected = theo_ratio * total_rounds
            obs_vs_exp = freq / expected if expected > 0 else 0
            writer.writerow([s, freq, round(freq/total_rounds*100, 2), round(freq/total_rounds, 4),
                           theo, round(theo_ratio*100, 4), round(obs_vs_exp, 2)])

    # ========== 3. 합계 구간별 분포 ==========
    range_freq = Counter(get_sum_range_label(r['sum']) for r in data)

    with open(OUTPUT_DIR / "sum_range_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['range', 'frequency', 'ratio', 'probability'])
        for rng in ['021-080', '081-120', '121-160', '161-200', '201-255']:
            freq = range_freq.get(rng, 0)
            writer.writerow([rng, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 4. 합계 끝자리 분석 ==========
    lastdigit_freq = Counter(r['sum'] % 10 for r in data)

    with open(OUTPUT_DIR / "sum_lastdigit_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['last_digit', 'frequency', 'ratio', 'probability'])
        for d in range(10):
            freq = lastdigit_freq.get(d, 0)
            writer.writerow([d, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 5. 홀수합/짝수합 분석 ==========
    oddeven_freq = Counter('홀' if r['sum'] % 2 == 1 else '짝' for r in data)

    with open(OUTPUT_DIR / "sum_oddeven_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'frequency', 'ratio', 'probability'])
        for t in ['홀', '짝']:
            freq = oddeven_freq.get(t, 0)
            writer.writerow([t, freq, round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 6. 최근 트렌드 분석 ==========
    recent_50 = data[-50:]
    recent_100 = data[-100:]

    recent_50_avg = statistics.mean(r['sum'] for r in recent_50)
    recent_100_avg = statistics.mean(r['sum'] for r in recent_100)

    with open(OUTPUT_DIR / "sum_trend_analysis.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['period', 'count', 'avg', 'std', 'min', 'max', 'median'])

        # 전체
        writer.writerow(['전체', total_rounds, round(sum_mean, 1), round(sum_std, 1),
                        min(sums), max(sums), sum_median])

        # 최근 100회
        r100_sums = [r['sum'] for r in recent_100]
        writer.writerow(['최근100회', 100, round(statistics.mean(r100_sums), 1),
                        round(statistics.stdev(r100_sums), 1), min(r100_sums), max(r100_sums),
                        statistics.median(r100_sums)])

        # 최근 50회
        r50_sums = [r['sum'] for r in recent_50]
        writer.writerow(['최근50회', 50, round(statistics.mean(r50_sums), 1),
                        round(statistics.stdev(r50_sums), 1), min(r50_sums), max(r50_sums),
                        statistics.median(r50_sums)])

    # ========== 7. 합계와 첫수/끝수 상관관계 ==========
    sum_first_corr = defaultdict(list)
    sum_last_corr = defaultdict(list)
    for r in data:
        sum_range = get_sum_range_label(r['sum'])
        sum_first_corr[sum_range].append(r['ball1'])
        sum_last_corr[sum_range].append(r['ball6'])

    with open(OUTPUT_DIR / "sum_firstend_correlation.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sum_range', 'count', 'first_avg', 'first_min', 'first_max', 'end_avg', 'end_min', 'end_max'])
        for rng in ['021-080', '081-120', '121-160', '161-200', '201-255']:
            firsts = sum_first_corr.get(rng, [])
            lasts = sum_last_corr.get(rng, [])
            if firsts:
                writer.writerow([rng, len(firsts),
                               round(statistics.mean(firsts), 1), min(firsts), max(firsts),
                               round(statistics.mean(lasts), 1), min(lasts), max(lasts)])

    # ========== 8. 연속 회차 합계 변화량 분석 ==========
    diff_freq = Counter()
    for i in range(1, len(data)):
        diff = data[i]['sum'] - data[i-1]['sum']
        diff_freq[diff] += 1

    with open(OUTPUT_DIR / "sum_diff_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['diff', 'frequency', 'ratio', 'probability'])
        for diff in sorted(diff_freq.keys()):
            freq = diff_freq[diff]
            writer.writerow([diff, freq, round(freq/(total_rounds-1)*100, 2), round(freq/(total_rounds-1), 4)])

    # 변화량 절대값 분석
    abs_diffs = [abs(data[i]['sum'] - data[i-1]['sum']) for i in range(1, len(data))]
    avg_abs_diff = statistics.mean(abs_diffs)

    # ========== 9. 10단위 구간 분석 ==========
    decade_freq = Counter((r['sum'] // 10) * 10 for r in data)

    with open(OUTPUT_DIR / "sum_decade_distribution.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['decade', 'range', 'frequency', 'ratio', 'probability'])
        for decade in sorted(decade_freq.keys()):
            freq = decade_freq[decade]
            writer.writerow([decade, f"{decade}-{decade+9}", freq,
                           round(freq/total_rounds*100, 1), round(freq/total_rounds, 4)])

    # ========== 10. 합계 빈도 상위 분석 ==========
    with open(OUTPUT_DIR / "sum_top_frequency.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'sum', 'frequency', 'ratio', 'rounds'])
        for rank, (s, freq) in enumerate(sum_freq.most_common(20), 1):
            rounds_list = [r['round'] for r in data if r['sum'] == s][-5:]  # 최근 5회차
            writer.writerow([rank, s, freq, round(freq/total_rounds*100, 2), ','.join(map(str, rounds_list))])

    # ========== Summary ==========
    with open(OUTPUT_DIR / "summary.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'description'])
        writer.writerow(['total_rounds', total_rounds, '분석 회차 수'])
        writer.writerow(['sum_avg', round(sum_mean, 2), '합계 평균'])
        writer.writerow(['sum_std', round(sum_std, 2), '합계 표준편차'])
        writer.writerow(['sum_median', sum_median, '합계 중앙값'])
        writer.writerow(['sum_q1', quartiles[0], '1사분위수'])
        writer.writerow(['sum_q3', quartiles[2], '3사분위수'])
        writer.writerow(['sum_min', min(sums), '합계 최소값'])
        writer.writerow(['sum_max', max(sums), '합계 최대값'])
        writer.writerow(['sum_mode', sum_freq.most_common(1)[0][0], '합계 최빈값'])
        writer.writerow(['sum_mode_freq', sum_freq.most_common(1)[0][1], '최빈값 출현 횟수'])
        writer.writerow(['odd_sum_ratio', round(oddeven_freq['홀']/total_rounds*100, 1), '홀수합 비율(%)'])
        writer.writerow(['range_121_160_ratio', round(range_freq['121-160']/total_rounds*100, 1), '121-160 구간 비율(%)'])
        writer.writerow(['recent_50_avg', round(recent_50_avg, 1), '최근 50회 평균'])
        writer.writerow(['avg_abs_diff', round(avg_abs_diff, 1), '회차간 평균 변화량(절대값)'])

    # ========== 출력 ==========
    print("=" * 70)
    print("Sum(합계) 전문가 인사이트")
    print("=" * 70)
    print(f"\n총 {total_rounds}회차 분석\n")

    print("[1. 합계 기본 통계]")
    print(f"    평균: {sum_mean:.1f}")
    print(f"    표준편차: {sum_std:.1f}")
    print(f"    중앙값: {sum_median}")
    print(f"    1사분위수(Q1): {quartiles[0]}")
    print(f"    3사분위수(Q3): {quartiles[2]}")
    print(f"    IQR(Q3-Q1): {quartiles[2] - quartiles[0]}")
    print(f"    최소: {min(sums)}, 최대: {max(sums)}")
    print(f"    최빈값: {sum_freq.most_common(1)[0][0]} ({sum_freq.most_common(1)[0][1]}회)")

    print("\n[2. 합계 빈도 상위 10개]")
    for s, c in sum_freq.most_common(10):
        theo = theoretical.get(s, 0)
        theo_ratio = theo / total_combos * 100
        print(f"    {s:3d}: {c:3d}회 ({c/total_rounds*100:5.1f}%) | 이론적 {theo_ratio:.2f}%")

    print("\n[3. 합계 구간별 분포]")
    for rng in ['021-080', '081-120', '121-160', '161-200', '201-255']:
        freq = range_freq.get(rng, 0)
        bar = '█' * int(freq/total_rounds*40)
        print(f"    {rng}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[4. 10단위 구간 분포]")
    for decade in sorted(decade_freq.keys()):
        freq = decade_freq[decade]
        if freq >= 10:  # 10회 이상만 표시
            bar = '█' * int(freq/total_rounds*30)
            print(f"    {decade:3d}-{decade+9:3d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[5. 합계 끝자리 분포]")
    for d in range(10):
        freq = lastdigit_freq.get(d, 0)
        bar = '█' * int(freq/total_rounds*25)
        print(f"    {d}: {freq:3d}회 ({freq/total_rounds*100:5.1f}%) {bar}")

    print("\n[6. 홀수합/짝수합 분포]")
    for t in ['홀', '짝']:
        freq = oddeven_freq.get(t, 0)
        print(f"    {t}수합: {freq:3d}회 ({freq/total_rounds*100:5.1f}%)")

    print("\n[7. 트렌드 분석]")
    print(f"    전체 평균: {sum_mean:.1f}")
    print(f"    최근 100회 평균: {recent_100_avg:.1f} ({'+' if recent_100_avg > sum_mean else ''}{recent_100_avg - sum_mean:.1f})")
    print(f"    최근 50회 평균: {recent_50_avg:.1f} ({'+' if recent_50_avg > sum_mean else ''}{recent_50_avg - sum_mean:.1f})")

    print("\n[8. 회차간 변화량]")
    print(f"    평균 변화량(절대값): {avg_abs_diff:.1f}")
    print(f"    최대 증가: +{max(data[i]['sum'] - data[i-1]['sum'] for i in range(1, len(data)))}")
    print(f"    최대 감소: {min(data[i]['sum'] - data[i-1]['sum'] for i in range(1, len(data)))}")

    print("\n[9. 합계-첫수/끝수 상관관계]")
    for rng in ['021-080', '081-120', '121-160', '161-200', '201-255']:
        firsts = sum_first_corr.get(rng, [])
        lasts = sum_last_corr.get(rng, [])
        if firsts:
            print(f"    {rng}: 첫수 평균 {statistics.mean(firsts):.1f}, 끝수 평균 {statistics.mean(lasts):.1f}")

    print("\n[10. 권장 합계 범위]")
    # Q1 ~ Q3 범위 (50% 확률 구간)
    print(f"    50% 확률 구간 (IQR): {quartiles[0]} ~ {quartiles[2]}")
    # 평균 ± 1σ (약 68% 확률 구간)
    print(f"    68% 확률 구간 (±1σ): {int(sum_mean - sum_std)} ~ {int(sum_mean + sum_std)}")
    # 평균 ± 2σ (약 95% 확률 구간)
    print(f"    95% 확률 구간 (±2σ): {int(sum_mean - 2*sum_std)} ~ {int(sum_mean + 2*sum_std)}")

    print("\n" + "=" * 70)
    print(f"CSV 파일 저장 완료: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    generate_insight()
