"""
조합 생성기 백테스트

과거 회차들에 대해 조합 생성기의 적중률을 검증
"""

import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import statistics

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
OUTPUT_DIR = Path(__file__).parent

# 소수 목록 (1~45)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_data():
    """당첨번호 데이터 로드"""
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


def get_recent_stats(data, n=50):
    """최근 N회차 통계"""
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


def get_range(n):
    if n <= 9: return 0
    elif n <= 19: return 1
    elif n <= 29: return 2
    elif n <= 39: return 3
    else: return 4


def is_prime(n):
    return n in PRIMES


def score_combination(balls, o_table):
    """조합에 대한 점수 계산"""
    score = 0

    ball1, ball2, ball3, ball4, ball5, ball6 = balls

    # 1. firstend
    if 1 <= ball1 <= 10:
        score += 10
    if 38 <= ball6 <= 45:
        score += 10

    # 2. second
    if 10 <= ball2 <= 19:
        score += 8
    elif 1 <= ball2 <= 9:
        score += 6

    if not is_prime(ball1) and not is_prime(ball2):
        score += 5

    # 3. third
    if 10 <= ball3 <= 19:
        score += 8
    elif 20 <= ball3 <= 29:
        score += 7

    # 4. fourth
    if 20 <= ball4 <= 29:
        score += 8
    elif 30 <= ball4 <= 39:
        score += 7

    prime_count_123 = sum(1 for b in [ball1, ball2, ball3] if is_prime(b))
    if prime_count_123 <= 1:
        score += 3

    # 5. fifth
    if 30 <= ball5 <= 39:
        score += 10
    elif 20 <= ball5 <= 29:
        score += 5

    sum14 = ball1 + ball2 + ball3 + ball4
    if sum14 <= 40 and 18 <= ball5 <= 33:
        score += 5
    elif 41 <= sum14 <= 70 and 22 <= ball5 <= 38:
        score += 5
    elif sum14 >= 71 and 30 <= ball5 <= 41:
        score += 5

    # 6. consecutive
    consecutive_pairs = 0
    for i in range(5):
        if balls[i+1] - balls[i] == 1:
            consecutive_pairs += 1

    if consecutive_pairs == 1:
        score += 8
    elif consecutive_pairs == 0:
        score += 5
    elif consecutive_pairs == 2:
        score += 3

    # 7. lastnum
    if 40 <= ball6 <= 45:
        score += 10
    elif 30 <= ball6 <= 39:
        score += 5

    # 8. sum
    total_sum = sum(balls)
    if 121 <= total_sum <= 160:
        score += 10
    elif 100 <= total_sum <= 170:
        score += 5

    # 9. range
    ranges_used = len(set(get_range(b) for b in balls))
    if ranges_used == 4:
        score += 8
    elif ranges_used == 5:
        score += 6
    elif ranges_used == 3:
        score += 4

    # 10. prime
    prime_count = sum(1 for b in balls if is_prime(b))
    if prime_count == 1 or prime_count == 2:
        score += 10
    elif prime_count == 3:
        score += 5

    # 11. shortcode
    ord_top24_count = 0
    for ball in balls:
        for ord_val, ball_val in o_table.items():
            if ball_val == ball and ord_val <= 24:
                ord_top24_count += 1
                break

    if 3 <= ord_top24_count <= 4:
        score += 10
    elif ord_top24_count == 5:
        score += 5

    # 12. onehot
    hot_bits = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
    cold_bits = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    hot_count = sum(1 for b in balls if b in hot_bits)
    cold_count = sum(1 for b in balls if b in cold_bits)

    if hot_count >= 2:
        score += hot_count * 2
    if cold_count >= 2:
        score -= cold_count

    return score


def generate_combinations_for_round(data, round_idx, max_combinations=500):
    """특정 회차에 대해 조합 생성 (해당 회차 이전 데이터만 사용)"""
    # 해당 회차 이전 데이터만 사용
    past_data = data[:round_idx]

    if len(past_data) < 50:
        return []

    # 최근 데이터에서 o_table 가져오기
    latest = past_data[-1]
    o_table = latest['o_table']

    # 최근 50회 통계
    pos_freq, all_freq = get_recent_stats(past_data, 50)

    # 후보 번호 생성
    candidates = {}

    # ball1: 1-10 범위
    candidates['ball1'] = []
    for num in range(1, 11):
        candidates['ball1'].append((num, pos_freq[0].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball1'].sort(key=lambda x: -x[1])

    # ball2: 1-25 범위
    candidates['ball2'] = []
    for num in range(2, 26):
        candidates['ball2'].append((num, pos_freq[1].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball2'].sort(key=lambda x: -x[1])

    # ball3: 5-35 범위
    candidates['ball3'] = []
    for num in range(5, 36):
        candidates['ball3'].append((num, pos_freq[2].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball3'].sort(key=lambda x: -x[1])

    # ball4: 15-42 범위
    candidates['ball4'] = []
    for num in range(15, 43):
        candidates['ball4'].append((num, pos_freq[3].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball4'].sort(key=lambda x: -x[1])

    # ball5: 25-44 범위
    candidates['ball5'] = []
    for num in range(25, 45):
        candidates['ball5'].append((num, pos_freq[4].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball5'].sort(key=lambda x: -x[1])

    # ball6: 30-45 범위
    candidates['ball6'] = []
    for num in range(30, 46):
        candidates['ball6'].append((num, pos_freq[5].get(num, 0) + all_freq.get(num, 0)))
    candidates['ball6'].sort(key=lambda x: -x[1])

    # 상위 후보만 사용
    b1_candidates = [x[0] for x in candidates['ball1'][:8]]
    b2_candidates = [x[0] for x in candidates['ball2'][:12]]
    b3_candidates = [x[0] for x in candidates['ball3'][:15]]
    b4_candidates = [x[0] for x in candidates['ball4'][:15]]
    b5_candidates = [x[0] for x in candidates['ball5'][:12]]
    b6_candidates = [x[0] for x in candidates['ball6'][:10]]

    # 조합 생성
    all_combos = []

    for b1 in b1_candidates:
        for b2 in b2_candidates:
            if b2 <= b1:
                continue
            for b3 in b3_candidates:
                if b3 <= b2:
                    continue
                for b4 in b4_candidates:
                    if b4 <= b3:
                        continue
                    for b5 in b5_candidates:
                        if b5 <= b4:
                            continue
                        for b6 in b6_candidates:
                            if b6 <= b5:
                                continue

                            balls = [b1, b2, b3, b4, b5, b6]
                            score = score_combination(balls, o_table)
                            all_combos.append({
                                'balls': balls,
                                'score': score
                            })

    # 점수순 정렬
    all_combos.sort(key=lambda x: -x['score'])

    return all_combos[:max_combinations]


def count_matches(predicted_balls, actual_balls):
    """맞춘 개수 계산"""
    return len(set(predicted_balls) & set(actual_balls))


def run_backtest(data, start_round=900, combo_count=500):
    """백테스트 실행"""
    results = []

    # 시작 회차 인덱스 찾기
    start_idx = None
    for i, d in enumerate(data):
        if d['round'] >= start_round:
            start_idx = i
            break

    if start_idx is None:
        print(f"Error: 시작 회차 {start_round}를 찾을 수 없습니다.")
        return []

    total_rounds = len(data) - start_idx

    print(f"백테스트 범위: {data[start_idx]['round']}회차 ~ {data[-1]['round']}회차 ({total_rounds}회)")
    print("=" * 80)

    for idx in range(start_idx, len(data)):
        current = data[idx]
        target_round = current['round']
        actual_balls = current['balls']

        # 조합 생성 (이전 데이터만 사용)
        combos = generate_combinations_for_round(data, idx, combo_count)

        if not combos:
            continue

        # 각 조합별 적중 수 계산
        match_counts = []
        for combo in combos:
            matches = count_matches(combo['balls'], actual_balls)
            match_counts.append(matches)

        # 최고 적중 수
        best_match = max(match_counts)

        # 각 적중 수별 조합 개수
        match_distribution = Counter(match_counts)

        # 상위 N개 조합의 평균 적중 수
        top10_avg = statistics.mean(match_counts[:10]) if len(match_counts) >= 10 else 0
        top50_avg = statistics.mean(match_counts[:50]) if len(match_counts) >= 50 else 0
        top100_avg = statistics.mean(match_counts[:100]) if len(match_counts) >= 100 else 0

        # 3개 이상 적중 조합 수
        match_3plus = sum(1 for m in match_counts if m >= 3)
        match_4plus = sum(1 for m in match_counts if m >= 4)
        match_5plus = sum(1 for m in match_counts if m >= 5)

        result = {
            'round': target_round,
            'actual_balls': actual_balls,
            'best_match': best_match,
            'top1_match': match_counts[0] if match_counts else 0,
            'top10_avg': round(top10_avg, 2),
            'top50_avg': round(top50_avg, 2),
            'top100_avg': round(top100_avg, 2),
            'match_3plus': match_3plus,
            'match_4plus': match_4plus,
            'match_5plus': match_5plus,
            'match_6': match_distribution.get(6, 0),
            'match_5': match_distribution.get(5, 0),
            'match_4': match_distribution.get(4, 0),
            'match_3': match_distribution.get(3, 0),
            'match_2': match_distribution.get(2, 0),
            'match_1': match_distribution.get(1, 0),
            'match_0': match_distribution.get(0, 0),
            'total_combos': len(combos)
        }

        results.append(result)

        # 진행상황 출력
        progress = (idx - start_idx + 1) / total_rounds * 100
        if (idx - start_idx + 1) % 50 == 0 or idx == len(data) - 1:
            print(f"진행: {idx - start_idx + 1}/{total_rounds} ({progress:.1f}%) - "
                  f"회차 {target_round}, 최고 적중: {best_match}개, 상위1위: {match_counts[0]}개, "
                  f"3+적중: {match_3plus}개")

    return results


def analyze_results(results):
    """결과 분석"""
    print("\n" + "=" * 80)
    print("백테스트 결과 분석")
    print("=" * 80)

    total = len(results)

    # 최고 적중 수 분포
    best_match_dist = Counter(r['best_match'] for r in results)
    print("\n[1. 최고 적중 수 분포 (500개 조합 중)]")
    for m in sorted(best_match_dist.keys(), reverse=True):
        count = best_match_dist[m]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"    {m}개 적중: {count:3d}회 ({pct:5.1f}%) {bar}")

    # 상위 1위 조합 적중 수 분포
    top1_dist = Counter(r['top1_match'] for r in results)
    print("\n[2. 상위 1위 조합 적중 수 분포]")
    for m in sorted(top1_dist.keys(), reverse=True):
        count = top1_dist[m]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"    {m}개 적중: {count:3d}회 ({pct:5.1f}%) {bar}")

    # 평균 적중 수
    avg_best = statistics.mean(r['best_match'] for r in results)
    avg_top1 = statistics.mean(r['top1_match'] for r in results)
    avg_top10 = statistics.mean(r['top10_avg'] for r in results)
    avg_top50 = statistics.mean(r['top50_avg'] for r in results)
    avg_top100 = statistics.mean(r['top100_avg'] for r in results)

    print("\n[3. 평균 적중 수]")
    print(f"    전체 최고: {avg_best:.2f}개")
    print(f"    상위 1위: {avg_top1:.2f}개")
    print(f"    상위 10개 평균: {avg_top10:.2f}개")
    print(f"    상위 50개 평균: {avg_top50:.2f}개")
    print(f"    상위 100개 평균: {avg_top100:.2f}개")

    # 3개 이상 적중 확률
    rounds_3plus = sum(1 for r in results if r['best_match'] >= 3)
    rounds_4plus = sum(1 for r in results if r['best_match'] >= 4)
    rounds_5plus = sum(1 for r in results if r['best_match'] >= 5)
    rounds_6 = sum(1 for r in results if r['best_match'] == 6)

    print("\n[4. N개 이상 적중 회차 비율]")
    print(f"    3개+: {rounds_3plus}/{total} ({rounds_3plus/total*100:.1f}%)")
    print(f"    4개+: {rounds_4plus}/{total} ({rounds_4plus/total*100:.1f}%)")
    print(f"    5개+: {rounds_5plus}/{total} ({rounds_5plus/total*100:.1f}%)")
    print(f"    6개 (1등): {rounds_6}/{total} ({rounds_6/total*100:.1f}%)")

    # 평균 3+ 적중 조합 수
    avg_3plus = statistics.mean(r['match_3plus'] for r in results)
    avg_4plus = statistics.mean(r['match_4plus'] for r in results)
    avg_5plus = statistics.mean(r['match_5plus'] for r in results)

    print("\n[5. 회차당 평균 N개+ 적중 조합 수 (500개 중)]")
    print(f"    3개+ 적중 조합: {avg_3plus:.1f}개")
    print(f"    4개+ 적중 조합: {avg_4plus:.1f}개")
    print(f"    5개+ 적중 조합: {avg_5plus:.1f}개")

    # 기대값 계산 (랜덤 대비)
    # 45C6 = 8,145,060
    # 6개 중 N개 맞출 확률
    print("\n[6. 랜덤 대비 성능]")
    random_avg = 6 * 6 / 45  # 약 0.8
    print(f"    랜덤 기대값: {random_avg:.2f}개")
    print(f"    시스템 평균: {avg_top1:.2f}개")
    print(f"    개선율: {(avg_top1 / random_avg - 1) * 100:.1f}%")

    return {
        'total_rounds': total,
        'avg_best_match': avg_best,
        'avg_top1_match': avg_top1,
        'avg_top10_match': avg_top10,
        'rounds_3plus_pct': rounds_3plus / total * 100,
        'rounds_4plus_pct': rounds_4plus / total * 100,
        'rounds_5plus_pct': rounds_5plus / total * 100,
        'avg_3plus_combos': avg_3plus,
        'avg_4plus_combos': avg_4plus
    }


def save_backtest_results(results, summary, output_path):
    """백테스트 결과 저장"""
    # 상세 결과 저장
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'round', 'actual_balls', 'best_match', 'top1_match',
            'top10_avg', 'top50_avg', 'top100_avg',
            'match_3plus', 'match_4plus', 'match_5plus',
            'match_6', 'match_5', 'match_4', 'match_3', 'match_2', 'match_1', 'match_0'
        ])

        for r in results:
            writer.writerow([
                r['round'],
                '-'.join(map(str, r['actual_balls'])),
                r['best_match'],
                r['top1_match'],
                r['top10_avg'],
                r['top50_avg'],
                r['top100_avg'],
                r['match_3plus'],
                r['match_4plus'],
                r['match_5plus'],
                r['match_6'],
                r['match_5'],
                r['match_4'],
                r['match_3'],
                r['match_2'],
                r['match_1'],
                r['match_0']
            ])

    print(f"\n결과 저장: {output_path}")

    # 요약 저장
    summary_path = output_path.parent / "backtest_summary.csv"
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in summary.items():
            writer.writerow([key, value])

    print(f"요약 저장: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='조합 생성기 백테스트')
    parser.add_argument('--start', type=int, default=900, help='시작 회차 (기본: 900)')
    parser.add_argument('--count', type=int, default=500, help='생성할 조합 수 (기본: 500)')
    args = parser.parse_args()

    print("=" * 80)
    print("조합 생성기 백테스트")
    print("=" * 80)

    # 데이터 로드
    data = load_data()
    print(f"\n총 데이터: {len(data)}회차 ({data[0]['round']} ~ {data[-1]['round']})")
    print(f"조합 수: {args.count}개")

    # 백테스트 실행
    results = run_backtest(data, args.start, args.count)

    if not results:
        print("Error: 백테스트 결과가 없습니다.")
        return

    # 결과 분석
    summary = analyze_results(results)

    # 결과 저장
    output_path = OUTPUT_DIR / "backtest_combination_results.csv"
    save_backtest_results(results, summary, output_path)

    print("\n" + "=" * 80)
    print("백테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
