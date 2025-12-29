"""
전체 파이프라인 백테스트

흐름:
1. 목표 회차까지 데이터 로드
2. 목표 회차-1까지 데이터로 인사이트 계산
3. 1_firstend → 2_146 → 3_235 순서로 예측
4. 당첨번호와 비교하여 결과 저장

실행:
    python backtest_full.py --start 900 --end 1000
    python backtest_full.py --round 1205  # 단일 회차
"""

import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
RESULT_DIR = BASE_DIR / "result"

# ============================================================
# 상수 정의 (fill_ord235.py 참조)
# ============================================================

# ord4 공식: A (0.60) + ±7 범위
FORMULA_RATIO = 0.60
OFFSET_RANGE = range(-7, 8)  # -7 ~ +7 (15개)

# 포지션별 최빈 구간
OPTIMAL_RANGES = {
    'ord2': (10, 19),
    'ord3': (10, 19),
    'ord5': (30, 39),
}

# 핫/콜드 비트
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

# 소수
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


# ============================================================
# 데이터 로드
# ============================================================

def load_all_data():
    """전체 당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            results.append({
                'round': int(row['round']),
                'balls': balls,
            })
    return sorted(results, key=lambda x: x['round'])


def get_data_until(all_data, target_round):
    """목표 회차 직전까지의 데이터만 반환"""
    return [r for r in all_data if r['round'] < target_round]


def get_winning_numbers(all_data, target_round):
    """목표 회차의 당첨번호 반환"""
    for r in all_data:
        if r['round'] == target_round:
            return r['balls']
    return None


# ============================================================
# 1_firstend: (ord1, ord6) 쌍 생성
# ============================================================

def generate_firstend_pairs(data):
    """(ord1, ord6) 쌍 빈도 계산 - 50/50 전략 적용"""
    # 1. 실제 출현한 쌍 집계
    seen_pairs = Counter()
    ord1_freq = Counter()
    ord6_freq = Counter()
    pair_rounds = defaultdict(list)

    for r in data:
        balls = r['balls']
        ord1, ord6 = balls[0], balls[5]
        seen_pairs[(ord1, ord6)] += 1
        ord1_freq[ord1] += 1
        ord6_freq[ord6] += 1
        pair_rounds[(ord1, ord6)].append(r['round'])

    # 2. seen 쌍: 상위 50% (빈도순)
    seen_sorted = sorted(seen_pairs.items(), key=lambda x: -x[1])
    seen_count = max(len(seen_sorted) // 2, 50)  # 최소 50개 보장
    seen_top = seen_sorted[:seen_count]

    # 3. unseen 쌍: 아직 안 나왔지만 가능성 높은 쌍
    unseen_scored = []
    for ord1 in range(1, 22):
        for ord6 in range(max(ord1 + 5, 23), 46):
            if (ord1, ord6) not in seen_pairs:
                score = ord1_freq.get(ord1, 0) + ord6_freq.get(ord6, 0)
                # 범위 보너스
                if 1 <= ord1 <= 10:
                    score += 20
                if 38 <= ord6 <= 45:
                    score += 20
                unseen_scored.append((ord1, ord6, score))

    unseen_sorted = sorted(unseen_scored, key=lambda x: -x[2])
    unseen_top = unseen_sorted[:seen_count]  # seen과 같은 수

    # 4. 합치기
    result = []
    for (o1, o6), freq in seen_top:
        result.append({
            'ord1': o1,
            'ord6': o6,
            'freq': freq,
            'rounds': pair_rounds.get((o1, o6), []),
            'type': 'seen',
        })
    for o1, o6, score in unseen_top:
        result.append({
            'ord1': o1,
            'ord6': o6,
            'freq': 0,
            'rounds': [],
            'type': 'unseen',
        })

    # 빈도순 정렬 (seen 우선)
    result.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord6']))
    return result


# ============================================================
# 2_146: ord4 계산
# ============================================================

def fill_ord4(pairs):
    """ord4 채우기 - 공식 A + ±7 범위"""
    rows = []
    seen = set()

    for pair in pairs:
        ord1 = pair['ord1']
        ord6 = pair['ord6']
        freq = pair['freq']
        rounds = pair['rounds']

        # 공식 A 기준값 계산
        base_ord4 = round(ord1 + (ord6 - ord1) * FORMULA_RATIO)

        # ±7 범위 적용
        for offset in OFFSET_RANGE:
            ord4 = base_ord4 + offset

            if ord4 <= ord1 or ord4 >= ord6:
                continue

            key = (ord1, ord4, ord6)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                'ord1': ord1,
                'ord4': ord4,
                'ord6': ord6,
                'freq': freq,
                'rounds': rounds,
                'offset': offset,
            })

    rows.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord4'], x['ord6']))
    return rows


# ============================================================
# 3_235: ord2, ord3, ord5 채우기
# ============================================================

def get_position_stats(data):
    """포지션별 빈도 통계"""
    pos_freq = {
        'ord2': Counter(),
        'ord3': Counter(),
        'ord5': Counter(),
    }
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


def score_candidate(num, pos_name, pos_freq, all_freq, recent_3):
    """후보 번호 점수 계산"""
    seen_numbers = set(pos_freq[pos_name].keys())
    optimal_min, optimal_max = OPTIMAL_RANGES[pos_name]

    if num in seen_numbers:
        score = pos_freq[pos_name][num] * 10
        if optimal_min <= num <= optimal_max:
            score += 15
    else:
        score = all_freq.get(num, 0) * 2
        if optimal_min <= num <= optimal_max:
            score += 20

    if num in HOT_BITS:
        score += 5
    if num in COLD_BITS:
        score -= 3
    if num in recent_3:
        score -= 5
    if num in PRIMES:
        score += 3

    return score


def find_top_candidates(candidates, pos_name, pos_freq, all_freq, recent_3, top_seen=2, top_unseen=1):
    """범위 내에서 Top-N seen + Top-M unseen 후보 선택"""
    if not candidates:
        return []

    seen_numbers = set(pos_freq[pos_name].keys())

    # seen/unseen 분리 및 점수 계산
    seen_scored = []
    unseen_scored = []

    for num in candidates:
        score = score_candidate(num, pos_name, pos_freq, all_freq, recent_3)
        if num in seen_numbers:
            seen_scored.append((num, score))
        else:
            unseen_scored.append((num, score))

    # 점수순 정렬
    seen_scored.sort(key=lambda x: -x[1])
    unseen_scored.sort(key=lambda x: -x[1])

    # Top-N seen + Top-M unseen
    result = [n for n, _ in seen_scored[:top_seen]]
    result += [n for n, _ in unseen_scored[:top_unseen]]

    return result if result else None


def fill_ord235(rows, pos_freq, all_freq, recent_3):
    """ord2, ord3, ord5 채우기 - Top-3 seen + Top-2 unseen 조합"""
    new_rows = []
    seen_combos = set()

    for row in rows:
        ord1 = row['ord1']
        ord4 = row['ord4']
        ord6 = row['ord6']

        # ord2 후보 (Top-3 seen + Top-2 unseen = 5개)
        ord2_range = list(range(ord1 + 1, ord4 - 1))
        ord2_picks = find_top_candidates(ord2_range, 'ord2', pos_freq, all_freq, recent_3, top_seen=3, top_unseen=2)
        if not ord2_picks:
            continue

        for ord2 in ord2_picks:
            # ord3 후보 (Top-3 seen + Top-2 unseen = 5개)
            ord3_range = list(range(ord2 + 1, ord4))
            ord3_picks = find_top_candidates(ord3_range, 'ord3', pos_freq, all_freq, recent_3, top_seen=3, top_unseen=2)
            if not ord3_picks:
                continue

            for ord3 in ord3_picks:
                # ord5 후보 (Top-3 seen + Top-2 unseen = 5개)
                ord5_range = list(range(ord4 + 1, ord6))
                ord5_picks = find_top_candidates(ord5_range, 'ord5', pos_freq, all_freq, recent_3, top_seen=3, top_unseen=2)
                if not ord5_picks:
                    continue

                for ord5 in ord5_picks:
                    if not (ord1 < ord2 < ord3 < ord4 < ord5 < ord6):
                        continue

                    # 중복 체크
                    combo = (ord1, ord2, ord3, ord4, ord5, ord6)
                    if combo in seen_combos:
                        continue
                    seen_combos.add(combo)

                    new_rows.append({
                        'ord1': ord1,
                        'ord2': ord2,
                        'ord3': ord3,
                        'ord4': ord4,
                        'ord5': ord5,
                        'ord6': ord6,
                        'freq': row['freq'],
                        'offset': row['offset'],
                    })

    new_rows.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord4'], x['ord6']))
    return new_rows


# ============================================================
# 백테스트 실행
# ============================================================

def run_backtest_single(all_data, target_round):
    """단일 회차 백테스트"""
    # 학습 데이터 (목표 회차 직전까지)
    train_data = get_data_until(all_data, target_round)
    if not train_data:
        return None

    # 당첨번호
    winning = get_winning_numbers(all_data, target_round)
    if winning is None:
        return None

    # 1. firstend
    pairs = generate_firstend_pairs(train_data)

    # 2. ord4
    rows_146 = fill_ord4(pairs)

    # 3. ord235
    pos_freq, all_freq, recent_3 = get_position_stats(train_data)
    predictions = fill_ord235(rows_146, pos_freq, all_freq, recent_3)

    if not predictions:
        return None

    # 당첨번호와 비교
    winning_set = set(winning)

    results = []
    for pred in predictions:
        pred_set = {pred['ord1'], pred['ord2'], pred['ord3'], pred['ord4'], pred['ord5'], pred['ord6']}
        match_count = len(pred_set & winning_set)
        matched_nums = sorted(pred_set & winning_set)

        results.append({
            'round': target_round,
            'prediction': [pred['ord1'], pred['ord2'], pred['ord3'], pred['ord4'], pred['ord5'], pred['ord6']],
            'winning': winning,
            'match_count': match_count,
            'matched_nums': matched_nums,
            'freq': pred['freq'],
            'offset': pred['offset'],
        })

    # 매치 수 내림차순 정렬
    results.sort(key=lambda x: -x['match_count'])

    return results


def run_backtest(start_round, end_round):
    """범위 백테스트"""
    all_data = load_all_data()

    # 유효한 회차 범위 확인
    rounds = [r['round'] for r in all_data]
    min_round, max_round = min(rounds), max(rounds)

    start_round = max(start_round, min_round + 1)  # 최소 1회차 학습 필요
    end_round = min(end_round, max_round)

    print(f"백테스트 범위: {start_round} ~ {end_round}")
    print(f"총 {end_round - start_round + 1}회차")
    print("-" * 60)

    all_results = []
    summary = {
        'total': 0,
        'match_0': 0, 'match_1': 0, 'match_2': 0,
        'match_3': 0, 'match_4': 0, 'match_5': 0, 'match_6': 0,
        'best_matches': [],
    }

    for target_round in range(start_round, end_round + 1):
        results = run_backtest_single(all_data, target_round)

        if results is None or len(results) == 0:
            continue

        summary['total'] += 1

        # 최고 매치 수
        best = results[0]
        match_count = best['match_count']
        summary[f'match_{match_count}'] += 1

        if match_count >= 4:
            summary['best_matches'].append({
                'round': target_round,
                'match_count': match_count,
                'prediction': best['prediction'],
                'winning': best['winning'],
                'matched': best['matched_nums'],
            })

        # 상세 결과 저장 (상위 10개만)
        for r in results[:10]:
            all_results.append({
                'round': r['round'],
                'ord1': r['prediction'][0],
                'ord2': r['prediction'][1],
                'ord3': r['prediction'][2],
                'ord4': r['prediction'][3],
                'ord5': r['prediction'][4],
                'ord6': r['prediction'][5],
                'winning': ','.join(map(str, r['winning'])),
                'match_count': r['match_count'],
                'matched_nums': ','.join(map(str, r['matched_nums'])),
                'freq': r['freq'],
                'offset': r['offset'],
            })

        # 진행 상황 출력
        if target_round % 50 == 0:
            print(f"  {target_round}회차 완료...")

    return all_results, summary


def save_results(all_results, summary, start_round, end_round):
    """결과 저장"""
    RESULT_DIR.mkdir(exist_ok=True)

    # 상세 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULT_DIR / f"backtest_{start_round}_{end_round}_{timestamp}.csv"

    with open(detail_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'round', 'ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6',
            'winning', 'match_count', 'matched_nums', 'freq', 'offset'
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n상세 결과 저장: {detail_path}")

    # 요약 출력
    print("\n" + "=" * 60)
    print("백테스트 요약")
    print("=" * 60)
    print(f"총 회차: {summary['total']}")
    print()
    print("매치 분포:")
    for i in range(7):
        count = summary[f'match_{i}']
        pct = count / summary['total'] * 100 if summary['total'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"  {i}개 적중: {count:4d}회 ({pct:5.1f}%) {bar}")

    # 4개 이상 적중
    match_4plus = sum(summary[f'match_{i}'] for i in range(4, 7))
    match_3plus = sum(summary[f'match_{i}'] for i in range(3, 7))
    print()
    print(f"3개+ 적중: {match_3plus}회 ({match_3plus/summary['total']*100:.1f}%)")
    print(f"4개+ 적중: {match_4plus}회 ({match_4plus/summary['total']*100:.1f}%)")

    if summary['best_matches']:
        print("\n[4개 이상 적중 회차]")
        for m in summary['best_matches']:
            print(f"  {m['round']}회: {m['match_count']}개 적중")
            print(f"    예측: {m['prediction']}")
            print(f"    당첨: {m['winning']}")
            print(f"    매치: {m['matched']}")


def main():
    parser = argparse.ArgumentParser(description='전체 파이프라인 백테스트')
    parser.add_argument('--start', type=int, default=900, help='시작 회차')
    parser.add_argument('--end', type=int, default=1200, help='종료 회차')
    parser.add_argument('--round', type=int, help='단일 회차 (start/end 무시)')
    args = parser.parse_args()

    if args.round:
        start_round = args.round
        end_round = args.round
    else:
        start_round = args.start
        end_round = args.end

    all_results, summary = run_backtest(start_round, end_round)
    save_results(all_results, summary, start_round, end_round)


if __name__ == "__main__":
    main()
