"""
당첨번호의 빈도 순위 분석

각 당첨번호가 최근 50회 빈도 기준으로 몇 위인지 분석
Top24 vs 나머지 분포 확인
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"


def load_data():
    """당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': sorted([int(row[f'ball{i}']) for i in range(1, 7)]),
            })
    return results


def get_frequency_ranks(data, idx, n=50):
    """최근 N회차 빈도 기준 전체 번호 순위"""
    past_data = data[max(0, idx-n):idx]

    freq = Counter()
    for r in past_data:
        for ball in r['balls']:
            freq[ball] += 1

    # 빈도순 정렬 (동점시 번호 오름차순)
    sorted_nums = sorted(range(1, 46), key=lambda x: (-freq.get(x, 0), x))

    # 순위 매핑
    ranks = {num: rank+1 for rank, num in enumerate(sorted_nums)}
    return ranks, freq


def analyze_winning_ranks(data, start_round=900):
    """당첨번호의 빈도 순위 분석"""

    # 시작 인덱스 찾기
    start_idx = None
    for i, d in enumerate(data):
        if d['round'] >= start_round:
            start_idx = i
            break

    if start_idx is None or start_idx < 50:
        print("데이터 부족")
        return

    # 결과 저장
    results = []

    # 각 포지션별 Top24 적중 수 집계
    pos_in_top24 = {i: 0 for i in range(6)}
    pos_in_rest = {i: 0 for i in range(6)}

    # 전체 Top24 개수별 분포
    top24_count_dist = Counter()

    for idx in range(start_idx, len(data)):
        current = data[idx]
        target_round = current['round']
        actual_balls = current['balls']

        # 이전 데이터 기준 빈도 순위
        ranks, freq = get_frequency_ranks(data, idx, 50)

        # 각 당첨번호의 순위
        ball_ranks = [ranks[b] for b in actual_balls]

        # Top24 개수
        in_top24 = sum(1 for r in ball_ranks if r <= 24)
        in_rest = 6 - in_top24

        top24_count_dist[in_top24] += 1

        # 포지션별 집계
        for pos, rank in enumerate(ball_ranks):
            if rank <= 24:
                pos_in_top24[pos] += 1
            else:
                pos_in_rest[pos] += 1

        results.append({
            'round': target_round,
            'balls': actual_balls,
            'ranks': ball_ranks,
            'in_top24': in_top24
        })

    total = len(results)

    # 출력
    print("=" * 80)
    print("당첨번호 빈도 순위 분석")
    print(f"분석 범위: {results[0]['round']}회차 ~ {results[-1]['round']}회차 ({total}회)")
    print("=" * 80)

    # 1. Top24 개수 분포
    print("\n[1. 당첨번호 중 Top24 개수 분포]")
    print("-" * 40)
    for cnt in sorted(top24_count_dist.keys(), reverse=True):
        freq = top24_count_dist[cnt]
        pct = freq / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {cnt}개: {freq:3d}회 ({pct:5.1f}%) {bar}")

    avg_top24 = sum(r['in_top24'] for r in results) / total
    print(f"\n  평균: {avg_top24:.2f}개 (6개 중)")

    # 2. 포지션별 Top24 비율
    print("\n[2. 포지션별 Top24 포함 비율]")
    print("-" * 40)
    for pos in range(6):
        in_top = pos_in_top24[pos]
        pct = in_top / total * 100
        print(f"  ball{pos+1}: {in_top:3d}/{total} ({pct:5.1f}%) Top24")

    # 3. 최근 10회 상세
    print("\n[3. 최근 10회 상세]")
    print("-" * 40)
    for r in results[-10:]:
        balls_str = '-'.join(f"{b:2d}" for b in r['balls'])
        ranks_str = '-'.join(f"{rk:2d}" for rk in r['ranks'])
        top24_marks = ''.join('●' if rk <= 24 else '○' for rk in r['ranks'])
        print(f"  {r['round']}회: {balls_str} | 순위: {ranks_str} | {top24_marks} ({r['in_top24']}개)")

    # 4. 순위 구간별 분포
    print("\n[4. 당첨번호 순위 구간별 분포]")
    print("-" * 40)

    all_ranks = []
    for r in results:
        all_ranks.extend(r['ranks'])

    rank_ranges = [
        (1, 10, "1-10위 (Top10)"),
        (11, 24, "11-24위 (Mid14)"),
        (25, 38, "25-38위 (Rest14)"),
        (39, 45, "39-45위 (Bottom7)")
    ]

    for start, end, label in rank_ranges:
        cnt = sum(1 for r in all_ranks if start <= r <= end)
        pct = cnt / len(all_ranks) * 100
        expected = (end - start + 1) / 45 * 100
        bar = '█' * int(pct / 2)
        print(f"  {label}: {cnt:4d}개 ({pct:5.1f}%) [기대값: {expected:.1f}%] {bar}")

    # 5. 포지션별 순위 평균
    print("\n[5. 포지션별 순위 평균]")
    print("-" * 40)

    for pos in range(6):
        pos_ranks = [r['ranks'][pos] for r in results]
        avg_rank = sum(pos_ranks) / len(pos_ranks)
        min_rank = min(pos_ranks)
        max_rank = max(pos_ranks)

        # Top24 비율
        top24_pct = sum(1 for r in pos_ranks if r <= 24) / len(pos_ranks) * 100

        print(f"  ball{pos+1}: 평균 {avg_rank:5.1f}위 (범위: {min_rank}-{max_rank}) | Top24: {top24_pct:.1f}%")

    # 6. 결론
    print("\n" + "=" * 80)
    print("[결론]")
    print("=" * 80)

    # Top24에서 3-4개 나오는 비율
    cnt_3_4 = top24_count_dist.get(3, 0) + top24_count_dist.get(4, 0)
    pct_3_4 = cnt_3_4 / total * 100

    # Top24에서 4-5개 나오는 비율
    cnt_4_5 = top24_count_dist.get(4, 0) + top24_count_dist.get(5, 0)
    pct_4_5 = cnt_4_5 / total * 100

    print(f"  - Top24에서 3-4개 나오는 비율: {pct_3_4:.1f}%")
    print(f"  - Top24에서 4-5개 나오는 비율: {pct_4_5:.1f}%")
    print(f"  - 평균 Top24 개수: {avg_top24:.2f}개")

    # 나머지(25위 이하) 분석
    rest_counts = [6 - r['in_top24'] for r in results]
    avg_rest = sum(rest_counts) / len(rest_counts)
    print(f"  - 평균 25위 이하(Rest) 개수: {avg_rest:.2f}개")

    return results


def main():
    data = load_data()
    print(f"총 데이터: {len(data)}회차")

    analyze_winning_ranks(data, start_round=900)


if __name__ == "__main__":
    main()
