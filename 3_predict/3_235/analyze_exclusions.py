"""
인사이트별 당첨번호 탈락 분석

각 인사이트 단계에서 실제 당첨번호가 얼마나 탈락되는지 분석
"""

import csv
from pathlib import Path
from collections import Counter, defaultdict
import statistics

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'ords': [int(row[f'ord{i}']) for i in range(1, 7)],
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
                'o_table': {i: int(row[f'o{i}']) for i in range(1, 46)}
            })
    return results


def get_range(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def is_prime(n):
    return n in PRIMES


def get_recent_stats(data, n=50):
    recent = data[-n:]
    pos_freq = {pos: Counter() for pos in range(6)}
    for r in recent:
        for pos, ball in enumerate(r['balls']):
            pos_freq[pos][ball] += 1
    all_freq = Counter()
    for r in recent:
        for ball in r['balls']:
            all_freq[ball] += 1
    return pos_freq, all_freq


def analyze_exclusions(data, start_round=900):
    """각 인사이트별 당첨번호 탈락 분석"""

    # 시작 인덱스 찾기
    start_idx = None
    for i, d in enumerate(data):
        if d['round'] >= start_round:
            start_idx = i
            break

    if start_idx is None:
        return None

    # 각 인사이트별 통계
    exclusion_stats = {
        'ball1_range': {'total': 0, 'excluded': 0, 'details': []},
        'ball2_range': {'total': 0, 'excluded': 0, 'details': []},
        'ball3_range': {'total': 0, 'excluded': 0, 'details': []},
        'ball4_range': {'total': 0, 'excluded': 0, 'details': []},
        'ball5_range': {'total': 0, 'excluded': 0, 'details': []},
        'ball6_range': {'total': 0, 'excluded': 0, 'details': []},
        'frequency_filter': {'total': 0, 'excluded': 0, 'details': []},
    }

    # 각 포지션별 후보 범위 (현재 사용 중인 범위)
    candidate_ranges = {
        'ball1': (1, 10),    # 1-10
        'ball2': (2, 25),    # 2-25
        'ball3': (5, 35),    # 5-35
        'ball4': (15, 42),   # 15-42
        'ball5': (25, 44),   # 25-44
        'ball6': (30, 45),   # 30-45
    }

    # 상세 분석
    exclusion_by_position = defaultdict(list)
    frequency_exclusion_details = []

    for idx in range(start_idx, len(data)):
        current = data[idx]
        past_data = data[:idx]

        if len(past_data) < 50:
            continue

        actual_balls = current['balls']
        target_round = current['round']

        # 최근 50회 통계
        pos_freq, all_freq = get_recent_stats(past_data, 50)

        # 1. 각 포지션별 범위 필터 분석
        for pos, (ball_key, (min_val, max_val)) in enumerate(candidate_ranges.items()):
            actual_ball = actual_balls[pos]
            exclusion_stats[f'{ball_key}_range']['total'] += 1

            if actual_ball < min_val or actual_ball > max_val:
                exclusion_stats[f'{ball_key}_range']['excluded'] += 1
                exclusion_stats[f'{ball_key}_range']['details'].append({
                    'round': target_round,
                    'actual': actual_ball,
                    'range': f'{min_val}-{max_val}'
                })
                exclusion_by_position[ball_key].append(actual_ball)

        # 2. 빈도 기반 필터 분석 (상위 N개만 후보로 선정)
        candidate_counts = [8, 12, 15, 15, 12, 10]  # 각 포지션별 후보 수

        for pos in range(6):
            ball_key = f'ball{pos+1}'
            actual_ball = actual_balls[pos]
            min_val, max_val = candidate_ranges[ball_key]

            # 해당 범위 내 번호들의 빈도순 정렬
            candidates = []
            for num in range(min_val, max_val + 1):
                freq = pos_freq[pos].get(num, 0) + all_freq.get(num, 0)
                candidates.append((num, freq))
            candidates.sort(key=lambda x: -x[1])

            # 상위 N개 후보
            top_n = candidate_counts[pos]
            top_candidates = set(x[0] for x in candidates[:top_n])

            exclusion_stats['frequency_filter']['total'] += 1
            if actual_ball not in top_candidates and min_val <= actual_ball <= max_val:
                exclusion_stats['frequency_filter']['excluded'] += 1

                # 해당 번호의 순위 찾기
                rank = None
                for r, (num, _) in enumerate(candidates):
                    if num == actual_ball:
                        rank = r + 1
                        break

                frequency_exclusion_details.append({
                    'round': target_round,
                    'position': ball_key,
                    'actual': actual_ball,
                    'rank': rank,
                    'top_n': top_n,
                    'candidates': [x[0] for x in candidates[:top_n]]
                })

    return exclusion_stats, exclusion_by_position, frequency_exclusion_details


def analyze_insight_conditions(data, start_round=900):
    """인사이트 조건별 당첨번호 적합도 분석"""

    start_idx = None
    for i, d in enumerate(data):
        if d['round'] >= start_round:
            start_idx = i
            break

    if start_idx is None:
        return None

    # 각 인사이트 조건별 통계
    insight_stats = defaultdict(lambda: {'match': 0, 'total': 0})

    for idx in range(start_idx, len(data)):
        current = data[idx]
        actual_balls = current['balls']
        ball1, ball2, ball3, ball4, ball5, ball6 = actual_balls

        # 1. firstend: ball1 범위
        insight_stats['firstend:ball1_1-10']['total'] += 1
        if 1 <= ball1 <= 10:
            insight_stats['firstend:ball1_1-10']['match'] += 1

        insight_stats['firstend:ball1_1-15']['total'] += 1
        if 1 <= ball1 <= 15:
            insight_stats['firstend:ball1_1-15']['match'] += 1

        # 2. firstend: ball6 범위
        insight_stats['firstend:ball6_38-45']['total'] += 1
        if 38 <= ball6 <= 45:
            insight_stats['firstend:ball6_38-45']['match'] += 1

        insight_stats['firstend:ball6_35-45']['total'] += 1
        if 35 <= ball6 <= 45:
            insight_stats['firstend:ball6_35-45']['match'] += 1

        insight_stats['firstend:ball6_30-45']['total'] += 1
        if 30 <= ball6 <= 45:
            insight_stats['firstend:ball6_30-45']['match'] += 1

        # 3. second: ball2 범위
        insight_stats['second:ball2_10-19']['total'] += 1
        if 10 <= ball2 <= 19:
            insight_stats['second:ball2_10-19']['match'] += 1

        insight_stats['second:ball2_1-25']['total'] += 1
        if 1 <= ball2 <= 25:
            insight_stats['second:ball2_1-25']['match'] += 1

        # 4. third: ball3 범위
        insight_stats['third:ball3_10-29']['total'] += 1
        if 10 <= ball3 <= 29:
            insight_stats['third:ball3_10-29']['match'] += 1

        insight_stats['third:ball3_5-35']['total'] += 1
        if 5 <= ball3 <= 35:
            insight_stats['third:ball3_5-35']['match'] += 1

        # 5. fourth: ball4 범위
        insight_stats['fourth:ball4_20-39']['total'] += 1
        if 20 <= ball4 <= 39:
            insight_stats['fourth:ball4_20-39']['match'] += 1

        insight_stats['fourth:ball4_15-42']['total'] += 1
        if 15 <= ball4 <= 42:
            insight_stats['fourth:ball4_15-42']['match'] += 1

        # 6. fifth: ball5 범위
        insight_stats['fifth:ball5_30-39']['total'] += 1
        if 30 <= ball5 <= 39:
            insight_stats['fifth:ball5_30-39']['match'] += 1

        insight_stats['fifth:ball5_25-44']['total'] += 1
        if 25 <= ball5 <= 44:
            insight_stats['fifth:ball5_25-44']['match'] += 1

        insight_stats['fifth:ball5_20-44']['total'] += 1
        if 20 <= ball5 <= 44:
            insight_stats['fifth:ball5_20-44']['match'] += 1

        # 7. lastnum: ball6 구간4 (40-45)
        insight_stats['lastnum:ball6_40-45']['total'] += 1
        if 40 <= ball6 <= 45:
            insight_stats['lastnum:ball6_40-45']['match'] += 1

        # 8. sum: 합계 범위
        total_sum = sum(actual_balls)
        insight_stats['sum:121-160']['total'] += 1
        if 121 <= total_sum <= 160:
            insight_stats['sum:121-160']['match'] += 1

        insight_stats['sum:100-170']['total'] += 1
        if 100 <= total_sum <= 170:
            insight_stats['sum:100-170']['match'] += 1

        insight_stats['sum:80-180']['total'] += 1
        if 80 <= total_sum <= 180:
            insight_stats['sum:80-180']['match'] += 1

        # 9. consecutive: 연속수
        consec_pairs = sum(1 for i in range(5) if actual_balls[i+1] - actual_balls[i] == 1)

        insight_stats['consecutive:0-1pair']['total'] += 1
        if consec_pairs <= 1:
            insight_stats['consecutive:0-1pair']['match'] += 1

        insight_stats['consecutive:0-2pair']['total'] += 1
        if consec_pairs <= 2:
            insight_stats['consecutive:0-2pair']['match'] += 1

        # 10. prime: 소수 개수
        prime_count = sum(1 for b in actual_balls if is_prime(b))

        insight_stats['prime:1-2']['total'] += 1
        if 1 <= prime_count <= 2:
            insight_stats['prime:1-2']['match'] += 1

        insight_stats['prime:0-3']['total'] += 1
        if 0 <= prime_count <= 3:
            insight_stats['prime:0-3']['match'] += 1

        # 11. range: 구간 사용 수
        ranges_used = len(set(get_range(b) for b in actual_balls))

        insight_stats['range:4segments']['total'] += 1
        if ranges_used == 4:
            insight_stats['range:4segments']['match'] += 1

        insight_stats['range:3-5segments']['total'] += 1
        if 3 <= ranges_used <= 5:
            insight_stats['range:3-5segments']['match'] += 1

    return insight_stats


def main():
    print("=" * 80)
    print("인사이트별 당첨번호 탈락 분석")
    print("=" * 80)

    data = load_data()
    print(f"\n총 데이터: {len(data)}회차")

    # 1. 범위 필터 및 빈도 필터 분석
    print("\n" + "=" * 80)
    print("1. 후보 생성 단계에서의 탈락 분석")
    print("=" * 80)

    exclusion_stats, exclusion_by_position, freq_details = analyze_exclusions(data, 900)

    print("\n[1-1. 범위 필터에 의한 탈락]")
    print(f"{'포지션':<12} {'전체':<8} {'탈락':<8} {'탈락률':<10} {'탈락 번호 예시'}")
    print("-" * 70)

    for key in ['ball1_range', 'ball2_range', 'ball3_range', 'ball4_range', 'ball5_range', 'ball6_range']:
        stats = exclusion_stats[key]
        rate = stats['excluded'] / stats['total'] * 100 if stats['total'] > 0 else 0

        # 탈락 번호 예시
        examples = exclusion_by_position[key.replace('_range', '')][:5]
        examples_str = ', '.join(map(str, examples)) if examples else '-'

        print(f"{key:<12} {stats['total']:<8} {stats['excluded']:<8} {rate:>6.1f}%    {examples_str}")

    print("\n[1-2. 빈도 필터에 의한 탈락 (상위 N개 후보 선정)]")
    stats = exclusion_stats['frequency_filter']
    rate = stats['excluded'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"전체: {stats['total']}, 탈락: {stats['excluded']}, 탈락률: {rate:.1f}%")

    # 포지션별 빈도 탈락 분석
    pos_freq_exclusion = defaultdict(list)
    for detail in freq_details:
        pos_freq_exclusion[detail['position']].append(detail)

    print(f"\n{'포지션':<10} {'탈락 횟수':<12} {'평균 순위':<12} {'후보 수'}")
    print("-" * 50)
    for pos in ['ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6']:
        details = pos_freq_exclusion[pos]
        if details:
            avg_rank = statistics.mean(d['rank'] for d in details if d['rank'])
            top_n = details[0]['top_n']
            print(f"{pos:<10} {len(details):<12} {avg_rank:>6.1f}       {top_n}")
        else:
            print(f"{pos:<10} {'0':<12} {'-':>6}       -")

    # 2. 인사이트 조건별 적합도 분석
    print("\n" + "=" * 80)
    print("2. 인사이트 조건별 당첨번호 적합도")
    print("=" * 80)

    insight_stats = analyze_insight_conditions(data, 900)

    print(f"\n{'인사이트 조건':<30} {'적합':<8} {'전체':<8} {'적합률':<10}")
    print("-" * 60)

    # 카테고리별로 정렬
    categories = ['firstend', 'second', 'third', 'fourth', 'fifth', 'lastnum', 'sum', 'consecutive', 'prime', 'range']

    for cat in categories:
        cat_items = [(k, v) for k, v in insight_stats.items() if k.startswith(cat)]
        for key, stats in sorted(cat_items):
            rate = stats['match'] / stats['total'] * 100 if stats['total'] > 0 else 0
            marker = "⚠️" if rate < 70 else "✅" if rate >= 90 else ""
            print(f"{key:<30} {stats['match']:<8} {stats['total']:<8} {rate:>6.1f}% {marker}")
        print()

    # 3. 문제가 되는 인사이트 요약
    print("=" * 80)
    print("3. 문제가 되는 인사이트 (적합률 80% 미만)")
    print("=" * 80)

    problematic = []
    for key, stats in insight_stats.items():
        rate = stats['match'] / stats['total'] * 100 if stats['total'] > 0 else 0
        if rate < 80:
            problematic.append((key, rate, stats['total'] - stats['match']))

    problematic.sort(key=lambda x: x[1])

    print(f"\n{'인사이트 조건':<30} {'적합률':<10} {'탈락 횟수'}")
    print("-" * 55)
    for key, rate, excluded in problematic:
        print(f"{key:<30} {rate:>6.1f}%    {excluded}회")

    # 4. 권장 조정사항
    print("\n" + "=" * 80)
    print("4. 권장 조정사항")
    print("=" * 80)

    print("""
[범위 필터 조정]
- ball1: 1-10 → 1-15 (ball1이 10 초과인 경우 커버)
- ball5: 25-44 → 20-44 (ball5가 25 미만인 경우 커버)
- ball6: 30-45 유지 (탈락률 낮음)

[빈도 후보 수 조정]
- ball1: 8개 → 10개
- ball2: 12개 → 15개
- ball3: 15개 → 18개
- ball4: 15개 → 18개
- ball5: 12개 → 15개
- ball6: 10개 → 12개

[점수 가중치 조정]
- 적합률 90%+ 인사이트: 가중치 유지 또는 증가
- 적합률 70-90% 인사이트: 가중치 유지
- 적합률 70% 미만 인사이트: 가중치 감소 또는 제거
""")

    # 결과 저장
    output_path = OUTPUT_DIR / "exclusion_analysis.csv"
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['insight', 'match', 'total', 'match_rate', 'excluded'])
        for key, stats in sorted(insight_stats.items()):
            rate = stats['match'] / stats['total'] * 100 if stats['total'] > 0 else 0
            writer.writerow([key, stats['match'], stats['total'], round(rate, 1), stats['total'] - stats['match']])

    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
