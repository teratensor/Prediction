"""
당첨번호 5-키 클러스터 크기 분석 (실시간 진행상황 표시)
"""
import csv
import time
import sys
from pathlib import Path
from collections import Counter, defaultdict

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
RESULT_DIR = Path(__file__).parent / "result"

# 상수
FORMULA_RATIO = 0.60
OFFSET_RANGE = range(-10, 11)


def load_all_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            results.append({'round': int(row['round']), 'balls': balls})
    return sorted(results, key=lambda x: x['round'])


def get_data_until(all_data, target_round):
    return [r for r in all_data if r['round'] < target_round]


def get_winning_numbers(all_data, target_round):
    for r in all_data:
        if r['round'] == target_round:
            return r['balls']
    return None


def load_insights_silent():
    return {
        'hot_bits': {29, 17, 27, 3, 25, 1, 19, 39, 4, 31},
        'cold_bits': {41, 43, 8, 14, 34, 26, 22, 44, 20, 7},
        'primes': {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43},
        'optimal_ranges': {
            'ord2': (10, 19), 'ord3': (15, 24),
            'ord5': (30, 39), 'ord6': (40, 45),
        },
    }


def get_position_stats(data):
    pos_freq = {'ord2': Counter(), 'ord3': Counter(), 'ord5': Counter()}
    all_freq = Counter()
    for r in data:
        balls = r['balls']
        pos_freq['ord2'][balls[1]] += 1
        pos_freq['ord3'][balls[2]] += 1
        pos_freq['ord5'][balls[4]] += 1
        for b in balls:
            all_freq[b] += 1
    recent_3 = set()
    for r in data[-3:]:
        recent_3.update(r['balls'])
    return pos_freq, all_freq, recent_3


def score_candidate(num, pos_name, pos_freq, all_freq, recent_3, insights):
    seen_numbers = set(pos_freq[pos_name].keys())
    optimal_min, optimal_max = insights['optimal_ranges'].get(pos_name, (1, 45))

    if num in seen_numbers:
        score = pos_freq[pos_name][num] * 10
        if optimal_min <= num <= optimal_max:
            score += 15
    else:
        score = all_freq.get(num, 0) * 2
        if optimal_min <= num <= optimal_max:
            score += 20

    if pos_name == 'ord5':
        if num in insights['hot_bits']: score += 8
        if num in insights['cold_bits']: score -= 5
    else:
        if num in insights['hot_bits']: score += 5
        if num in insights['cold_bits']: score -= 3

    if num in recent_3: score -= 5
    if pos_name != 'ord5' and num in insights['primes']: score += 3

    return score


def find_top_candidates(candidates, pos_name, pos_freq, all_freq, recent_3, insights, top_n=15):
    if not candidates:
        return []
    scored = [(num, score_candidate(num, pos_name, pos_freq, all_freq, recent_3, insights)) for num in candidates]
    scored.sort(key=lambda x: -x[1])
    return [num for num, _ in scored[:top_n]]


def generate_cluster_counts(all_data, target_round):
    """예측 조합 생성 및 클러스터 카운트"""
    train_data = get_data_until(all_data, target_round)
    if len(train_data) == 0:
        return Counter()

    insights = load_insights_silent()
    pos_freq, all_freq, recent_3 = get_position_stats(train_data)

    # firstend 쌍
    pair_rounds = defaultdict(list)
    for r in train_data:
        balls = r['balls']
        pair_rounds[(balls[0], balls[5])].append(r['round'])

    cluster_counts = Counter()

    for ord1 in range(1, 22):
        for ord6 in range(max(ord1 + 5, 23), 46):
            base_ord4 = round(ord1 + (ord6 - ord1) * FORMULA_RATIO)

            for offset in OFFSET_RANGE:
                ord4 = base_ord4 + offset
                if ord4 <= ord1 or ord4 >= ord6:
                    continue

                ord2_candidates = list(range(ord1 + 1, ord4 - 1))
                ord2_picks = find_top_candidates(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3, insights, 15)
                if not ord2_picks:
                    continue

                ord5_candidates = list(range(ord4 + 1, ord6))
                ord5_picks = find_top_candidates(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3, insights, 15)
                if not ord5_picks:
                    continue

                for ord2 in ord2_picks:
                    ord3_candidates = list(range(ord2 + 1, ord4))
                    ord3_picks = find_top_candidates(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3, insights, 15)
                    if not ord3_picks:
                        continue

                    for ord3 in ord3_picks:
                        for ord5 in ord5_picks:
                            if not (ord1 < ord2 < ord3 < ord4 < ord5 < ord6):
                                continue

                            combo = (ord1, ord2, ord3, ord4, ord5, ord6)
                            for i in range(6):
                                key = combo[:i] + combo[i+1:]
                                cluster_counts[key] += 1

    return cluster_counts


def format_time(seconds):
    """초를 mm:ss 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.0f}초"
    else:
        m, s = divmod(int(seconds), 60)
        return f"{m}분 {s}초"


def main():
    print("=" * 60)
    print("당첨번호 5-키 클러스터 크기 분석")
    print("=" * 60)

    all_data = load_all_data()
    print(f"데이터: {len(all_data)}회차\n")

    # 분석 범위
    start_round = 900
    end_round = 1204
    rounds_to_analyze = [r['round'] for r in all_data if start_round <= r['round'] <= end_round]
    total_rounds = len(rounds_to_analyze)

    print(f"분석 범위: {start_round} ~ {end_round} ({total_rounds}회차)")
    print("-" * 60)

    # 결과 저장
    all_sizes = []
    not_in_db_count = 0
    results = []

    start_time = time.time()

    for idx, target_round in enumerate(rounds_to_analyze):
        round_start = time.time()

        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        # 클러스터 카운트 생성
        cluster_counts = generate_cluster_counts(all_data, target_round)

        # 당첨번호 6개의 5-키 크기 확인
        winning_5keys = [tuple(winning[:i] + winning[i+1:]) for i in range(6)]
        sizes = [cluster_counts.get(key, 0) for key in winning_5keys]

        # 통계 수집
        nonzero_sizes = [s for s in sizes if s > 0]
        if nonzero_sizes:
            all_sizes.extend(nonzero_sizes)

        in_db = sum(1 for s in sizes if s > 0)
        not_in_db_count += (6 - in_db)

        results.append({
            'round': target_round,
            'winning': winning,
            'sizes': sizes,
            'in_db': in_db,
        })

        # 진행상황 계산
        round_time = time.time() - round_start
        elapsed = time.time() - start_time
        completed = idx + 1
        remaining = total_rounds - completed

        if completed > 0:
            avg_time = elapsed / completed
            eta = avg_time * remaining
        else:
            eta = 0

        # 실시간 출력
        size_str = ",".join(str(s) for s in sizes)
        progress_pct = completed / total_rounds * 100

        sys.stdout.write(f"\r[{completed:3d}/{total_rounds}] {progress_pct:5.1f}% | "
                        f"{target_round}회차: {in_db}/6 in DB, sizes=[{size_str}] | "
                        f"경과: {format_time(elapsed)}, 남은시간: {format_time(eta)}    ")
        sys.stdout.flush()

    print("\n" + "=" * 60)

    # 최종 통계
    total_time = time.time() - start_time
    total_5keys = len(results) * 6

    print(f"\n[분석 완료]")
    print(f"  총 소요시간: {format_time(total_time)}")
    print(f"  분석 회차: {len(results)}개")
    print(f"  총 5-키: {total_5keys}개")
    print(f"  DB에 없음: {not_in_db_count}개 ({not_in_db_count/total_5keys*100:.1f}%)")
    print(f"  DB에 있음: {len(all_sizes)}개 ({len(all_sizes)/total_5keys*100:.1f}%)")

    if all_sizes:
        print(f"\n[당첨 5-키 크기 통계]")
        print(f"  최소: {min(all_sizes)}")
        print(f"  최대: {max(all_sizes)}")
        print(f"  평균: {sum(all_sizes)/len(all_sizes):.1f}")

        # 크기 분포
        size_dist = Counter(all_sizes)
        print(f"\n[크기별 분포]")

        ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 50), (51, 100)]
        for low, high in ranges:
            count = sum(1 for s in all_sizes if low <= s <= high)
            pct = count / len(all_sizes) * 100
            bar = '█' * int(pct / 2)
            print(f"  {low:3d}-{high:3d}: {count:4d}개 ({pct:5.1f}%) {bar}")

        # 상위 빈도 크기
        print(f"\n[가장 빈번한 크기 Top 10]")
        for size, freq in size_dist.most_common(10):
            pct = freq / len(all_sizes) * 100
            print(f"  크기 {size:3d}: {freq:4d}개 ({pct:4.1f}%)")

    # CSV 저장
    RESULT_DIR.mkdir(exist_ok=True)
    result_path = RESULT_DIR / "winning_5key_sizes.csv"

    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'in_db'])
        for r in results:
            writer.writerow([
                r['round'],
                str(r['winning']),
                *r['sizes'],
                r['in_db'],
            ])

    print(f"\n[저장] {result_path}")


if __name__ == "__main__":
    main()
