"""
왜 6개 일치를 못하는가? - 단계별 병목 분석

main.py 파이프라인 (현재 설정):
1. FirstEnd: (ord1, ord6) 쌍 477개 생성
2. ord4: 공식 A (0.60) + ±7 범위 (15개 offset)
3. ord235: 각 포지션 점수 Top-5 선택

이 스크립트는 각 단계에서 실제 당첨번호가 탈락하는 비율을 분석하고
개선 방안을 제시합니다.

실행:
    python 4_backtest/why.py
    python 4_backtest/why.py --start 900 --end 1000
"""

import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
INSIGHTS_DIR = BASE_DIR / "2_insights"

# 현재 main.py 상수
FORMULA_RATIO = 0.60
OFFSET_RANGE = range(-10, 11)  # -10 ~ +10 (21개)
TOP_N = 15  # 현재 main.py 설정


def load_insights():
    """인사이트 동적 로드 (main.py와 동일)"""
    insights = {
        'hot_bits': set(),
        'cold_bits': set(),
        'primes': set(),
        'optimal_ranges': {},
    }

    # HOT/COLD bits
    hot_cold_path = INSIGHTS_DIR / "7_onehot" / "statistics" / "hot_cold_bits.csv"
    if hot_cold_path.exists():
        with open(hot_cold_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['category'] == 'HOT':
                    insights['hot_bits'].add(int(row['bit']))
                elif row['category'] == 'COLD':
                    insights['cold_bits'].add(int(row['bit']))
    else:
        insights['hot_bits'] = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
        insights['cold_bits'] = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    # 소수
    prime_path = INSIGHTS_DIR / "5_prime" / "statistics" / "prime_frequency.csv"
    if prime_path.exists():
        with open(prime_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                insights['primes'].add(int(row['prime']))
    else:
        insights['primes'] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

    # 최빈 구간 로드 (ord2, ord3, ord5, ord6)
    range_path = INSIGHTS_DIR / "4_range" / "statistics" / "position_range_distribution.csv"
    if range_path.exists():
        best_range = {}
        with open(range_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos = int(row['position'])
                prob = float(row['probability'])
                if pos in [2, 3, 5, 6]:
                    if pos not in best_range or prob > best_range[pos][1]:
                        range_idx = int(row['range'])
                        best_range[pos] = (range_idx, prob)
        range_map = {0: (1, 9), 1: (10, 19), 2: (20, 29), 3: (30, 39), 4: (40, 45)}
        for pos, (range_idx, _) in best_range.items():
            insights['optimal_ranges'][f'ord{pos}'] = range_map[range_idx]
        # ord3 범위 조정: 실제 평균 19.4이므로 (15, 24)가 더 적합
        if 'ord3' in insights['optimal_ranges']:
            insights['optimal_ranges']['ord3'] = (15, 24)
    else:
        # 8개 인사이트 분석 기반 최적 범위
        insights['optimal_ranges'] = {
            'ord2': (10, 19),   # 48.3%
            'ord3': (15, 24),   # 평균 19.4 기반 조정
            'ord5': (30, 39),   # 54.6%
            'ord6': (40, 45),   # 57.8%
        }

    return insights


INSIGHTS = load_insights()


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
    return [r for r in all_data if r['round'] < target_round]


def get_winning_numbers(all_data, target_round):
    for r in all_data:
        if r['round'] == target_round:
            return r['balls']
    return None


# ============================================================
# 1단계 분석: FirstEnd - (ord1, ord6) 쌍
# ============================================================

def analyze_stage1_firstend(all_data, start_round, end_round):
    """1단계: (ord1, ord6) 쌍이 생성된 477개에 포함되는지"""
    results = {
        'total': 0,
        'pass': 0,
        'fail': 0,
        'fail_details': [],
        'ord1_dist': Counter(),
        'ord6_dist': Counter(),
    }

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        results['total'] += 1
        actual_ord1, actual_ord6 = winning[0], winning[5]

        results['ord1_dist'][actual_ord1] += 1
        results['ord6_dist'][actual_ord6] += 1

        # 477개 쌍 범위: ord1 <= 21, ord6 >= 23, 최소 5칸 차이
        valid = (1 <= actual_ord1 <= 21 and
                 actual_ord6 >= max(actual_ord1 + 5, 23) and
                 actual_ord6 <= 45)

        if valid:
            results['pass'] += 1
        else:
            results['fail'] += 1
            results['fail_details'].append({
                'round': target_round,
                'ord1': actual_ord1,
                'ord6': actual_ord6,
                'reason': f"ord1={actual_ord1}, ord6={actual_ord6} - 범위 밖"
            })

    return results


# ============================================================
# 2단계 분석: ord4 - 공식 A + ±7 범위
# ============================================================

def analyze_stage2_ord4(all_data, start_round, end_round):
    """2단계: ord4가 공식 A ± 7 범위에 포함되는지"""
    results = {
        'total': 0,
        'pass': 0,
        'fail': 0,
        'fail_details': [],
        'offset_dist': Counter(),
        'offset_all': Counter(),  # 범위 밖 포함 전체 분포
    }

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        results['total'] += 1
        ord1, ord4, ord6 = winning[0], winning[3], winning[5]

        base_ord4 = round(ord1 + (ord6 - ord1) * FORMULA_RATIO)
        actual_offset = ord4 - base_ord4

        results['offset_all'][actual_offset] += 1

        if actual_offset in OFFSET_RANGE:
            results['pass'] += 1
            results['offset_dist'][actual_offset] += 1
        else:
            results['fail'] += 1
            results['fail_details'].append({
                'round': target_round,
                'ord1': ord1,
                'ord4': ord4,
                'ord6': ord6,
                'base_ord4': base_ord4,
                'actual_offset': actual_offset,
            })

    return results


# ============================================================
# 3단계 분석: ord235 - 점수 기반 Top-N 선택
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
    """후보 번호 점수 계산 - 8개 인사이트 활용, 포지션별 차별화 (main.py와 동일)

    [적용 인사이트]
    - 4_range: 포지션별 최빈 구간 보너스
    - 5_prime: 소수 보너스 (ord5 제외 - ball5 소수 18.5%로 낮음)
    - 7_onehot: HOT/COLD 비트 (ord5는 HOT +8, COLD -5로 강화)
    """
    seen_numbers = set(pos_freq[pos_name].keys())
    optimal_min, optimal_max = INSIGHTS['optimal_ranges'].get(pos_name, (1, 45))

    # 기본 점수: 빈도 기반
    if num in seen_numbers:
        score = pos_freq[pos_name][num] * 10
        if optimal_min <= num <= optimal_max:
            score += 15
    else:
        score = all_freq.get(num, 0) * 2
        if optimal_min <= num <= optimal_max:
            score += 20

    # [7. onehot] HOT/COLD - 포지션별 차별화
    # 6매치에서 ord5 HOT 비율이 44.4%로 높음 → 보너스 강화
    if pos_name == 'ord5':
        if num in INSIGHTS['hot_bits']:
            score += 8  # 기존 5 → 8 (강화)
        if num in INSIGHTS['cold_bits']:
            score -= 5  # 기존 3 → 5 (강화)
    else:
        if num in INSIGHTS['hot_bits']:
            score += 5
        if num in INSIGHTS['cold_bits']:
            score -= 3

    # 최근 3회 출현 페널티
    if num in recent_3:
        score -= 5

    # [5. prime] 소수 보너스 - ord5 제외
    # ball5 소수 비율이 18.5%로 낮음 → ord5는 소수 보너스 제거
    if pos_name != 'ord5':
        if num in INSIGHTS['primes']:
            score += 3

    return score


def get_top_candidates(candidates, pos_name, pos_freq, all_freq, recent_3, top_n):
    """범위 내에서 상위 N개 후보 선택"""
    if not candidates:
        return []

    scored = []
    for num in candidates:
        score = score_candidate(num, pos_name, pos_freq, all_freq, recent_3)
        scored.append((num, score))

    scored.sort(key=lambda x: -x[1])
    return [num for num, _ in scored[:top_n]]


def analyze_stage3_ord235(all_data, start_round, end_round, top_n=TOP_N):
    """3단계: ord2, ord3, ord5가 Top-N에 포함되는지 분석"""
    results = {
        'total': 0,
        'ord2_pass': 0,
        'ord3_pass': 0,
        'ord5_pass': 0,
        'all_pass': 0,
        'ord2_fail_details': [],
        'ord3_fail_details': [],
        'ord5_fail_details': [],
        'rank_dist': {'ord2': Counter(), 'ord3': Counter(), 'ord5': Counter()},
        'top_n': top_n,
    }

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        train_data = get_data_until(all_data, target_round)
        if len(train_data) == 0:
            continue

        results['total'] += 1

        ord1, ord2, ord3, ord4, ord5, ord6 = winning
        pos_freq, all_freq, recent_3 = get_position_stats(train_data)

        ord2_pass = False
        ord3_pass = False
        ord5_pass = False

        # ord2 분석
        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        if ord2 in ord2_candidates:
            scores = [(n, score_candidate(n, 'ord2', pos_freq, all_freq, recent_3)) for n in ord2_candidates]
            scores.sort(key=lambda x: -x[1])
            rank = next(i for i, (n, _) in enumerate(scores, 1) if n == ord2)
            results['rank_dist']['ord2'][rank] += 1

            if rank <= top_n:
                results['ord2_pass'] += 1
                ord2_pass = True
            else:
                results['ord2_fail_details'].append({
                    'round': target_round,
                    'actual': ord2,
                    'rank': rank,
                    'candidates': len(ord2_candidates),
                })
        else:
            results['ord2_fail_details'].append({
                'round': target_round,
                'actual': ord2,
                'rank': 'N/A',
                'reason': 'out of range',
            })

        # ord3 분석 (실제 ord2 기준)
        ord3_candidates = list(range(ord2 + 1, ord4))
        if ord3 in ord3_candidates:
            scores = [(n, score_candidate(n, 'ord3', pos_freq, all_freq, recent_3)) for n in ord3_candidates]
            scores.sort(key=lambda x: -x[1])
            rank = next(i for i, (n, _) in enumerate(scores, 1) if n == ord3)
            results['rank_dist']['ord3'][rank] += 1

            if rank <= top_n:
                results['ord3_pass'] += 1
                ord3_pass = True
            else:
                results['ord3_fail_details'].append({
                    'round': target_round,
                    'actual': ord3,
                    'rank': rank,
                })
        else:
            results['ord3_fail_details'].append({
                'round': target_round,
                'actual': ord3,
                'rank': 'N/A',
                'reason': 'out of range',
            })

        # ord5 분석
        ord5_candidates = list(range(ord4 + 1, ord6))
        if ord5 in ord5_candidates:
            scores = [(n, score_candidate(n, 'ord5', pos_freq, all_freq, recent_3)) for n in ord5_candidates]
            scores.sort(key=lambda x: -x[1])
            rank = next(i for i, (n, _) in enumerate(scores, 1) if n == ord5)
            results['rank_dist']['ord5'][rank] += 1

            if rank <= top_n:
                results['ord5_pass'] += 1
                ord5_pass = True
            else:
                results['ord5_fail_details'].append({
                    'round': target_round,
                    'actual': ord5,
                    'rank': rank,
                })
        else:
            results['ord5_fail_details'].append({
                'round': target_round,
                'actual': ord5,
                'rank': 'N/A',
                'reason': 'out of range',
            })

        # 모두 통과?
        if ord2_pass and ord3_pass and ord5_pass:
            results['all_pass'] += 1

    return results


# ============================================================
# 종합 분석: 6개 일치 가능 회차 분석
# ============================================================

def analyze_full_pipeline(all_data, start_round, end_round, top_n=TOP_N):
    """전체 파이프라인 시뮬레이션 - 6개 일치 가능 여부"""
    results = {
        'total': 0,
        'stage1_fail': 0,
        'stage2_fail': 0,
        'stage3_fail': 0,
        'success': 0,
        'success_rounds': [],
        'fail_by_position': Counter(),
        '5match_miss_position': Counter(),  # 5개 적중 시 놓친 포지션
    }

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        train_data = get_data_until(all_data, target_round)
        if len(train_data) == 0:
            continue

        results['total'] += 1
        ord1, ord2, ord3, ord4, ord5, ord6 = winning

        # 1단계: firstend 체크
        valid_firstend = (1 <= ord1 <= 21 and
                          ord6 >= max(ord1 + 5, 23) and
                          ord6 <= 45)
        if not valid_firstend:
            results['stage1_fail'] += 1
            results['fail_by_position']['ord1/ord6 범위'] += 1
            continue

        # 2단계: ord4 체크
        base_ord4 = round(ord1 + (ord6 - ord1) * FORMULA_RATIO)
        actual_offset = ord4 - base_ord4
        if actual_offset not in OFFSET_RANGE:
            results['stage2_fail'] += 1
            results['fail_by_position']['ord4 범위'] += 1
            continue

        # 3단계: ord235 체크 (Top-N)
        pos_freq, all_freq, recent_3 = get_position_stats(train_data)

        fail_positions = []

        # ord2 (실제 ord1, ord4 기준 범위)
        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        top_ord2 = get_top_candidates(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3, top_n)
        if ord2 not in top_ord2:
            fail_positions.append('ord2')

        # ord3 (실제 ord2, ord4 기준 범위)
        ord3_candidates = list(range(ord2 + 1, ord4))
        top_ord3 = get_top_candidates(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3, top_n)
        if ord3 not in top_ord3:
            fail_positions.append('ord3')

        # ord5 (실제 ord4, ord6 기준 범위)
        ord5_candidates = list(range(ord4 + 1, ord6))
        top_ord5 = get_top_candidates(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3, top_n)
        if ord5 not in top_ord5:
            fail_positions.append('ord5')

        if fail_positions:
            results['stage3_fail'] += 1
            for pos in fail_positions:
                results['fail_by_position'][pos] += 1

            # 5개 적중 가능한 경우 (실패 1개만)
            if len(fail_positions) == 1:
                results['5match_miss_position'][fail_positions[0]] += 1
        else:
            results['success'] += 1
            results['success_rounds'].append(target_round)

    return results


# ============================================================
# Top-N 비교 분석
# ============================================================

def compare_top_n(all_data, start_round, end_round):
    """다양한 Top-N 값에 대한 비교 분석"""
    results = {}

    # 실제 측정된 조합수 (main.py 기준)
    actual_combos = {
        1: 6206,
        3: 142338,
        5: 562301,
        7: 1277979,
        10: 2700717,
        15: 5500000,  # 추정값
    }

    for top_n in [1, 3, 5, 7, 10, 15]:
        stage3 = analyze_stage3_ord235(all_data, start_round, end_round, top_n)
        full = analyze_full_pipeline(all_data, start_round, end_round, top_n)
        combo_count = actual_combos.get(top_n, 0)
        results[top_n] = {
            'ord2_pass': stage3['ord2_pass'],
            'ord3_pass': stage3['ord3_pass'],
            'ord5_pass': stage3['ord5_pass'],
            'all_pass': stage3['all_pass'],
            'success': full['success'],
            'total': stage3['total'],
            'combinations': f"~{combo_count:,}",
        }

    return results


def print_report(stage1, stage2, stage3, full, top_n_compare, start_round, end_round):
    """분석 결과 출력"""
    print("=" * 70)
    print(f"왜 6개 일치를 못하는가? - 병목 분석 (main.py 현재 설정)")
    print(f"분석 범위: {start_round} ~ {end_round}회차 ({stage1['total']}개)")
    print(f"현재 설정: offset=±7, top_n={TOP_N}")
    print("=" * 70)

    # 1단계: FirstEnd
    print(f"\n{'=' * 70}")
    print(f"[1단계: FirstEnd] - (ord1, ord6) 쌍 477개 생성")
    print(f"{'=' * 70}")
    print(f"  통과: {stage1['pass']:4d}회 ({stage1['pass']/stage1['total']*100:.1f}%)")
    print(f"  실패: {stage1['fail']:4d}회 ({stage1['fail']/stage1['total']*100:.1f}%)")

    # ord1 분포
    print(f"\n  [ord1 분포 (상위 5개)]")
    for val, count in stage1['ord1_dist'].most_common(5):
        pct = count / stage1['total'] * 100
        bar = '█' * int(pct / 2)
        print(f"    {val:2d}: {count:3d}회 ({pct:5.1f}%) {bar}")

    # ord6 분포
    print(f"\n  [ord6 분포 (상위 5개)]")
    for val, count in stage1['ord6_dist'].most_common(5):
        pct = count / stage1['total'] * 100
        bar = '█' * int(pct / 2)
        print(f"    {val:2d}: {count:3d}회 ({pct:5.1f}%) {bar}")

    if stage1['fail'] > 0:
        print(f"\n  [실패 사례]")
        for detail in stage1['fail_details'][:3]:
            print(f"    {detail['round']}회: {detail['reason']}")

    # 2단계: ord4
    print(f"\n{'=' * 70}")
    print(f"[2단계: ord4] - 공식 A (0.60) + ±7 범위")
    print(f"{'=' * 70}")
    print(f"  통과: {stage2['pass']:4d}회 ({stage2['pass']/stage2['total']*100:.1f}%)")
    print(f"  실패: {stage2['fail']:4d}회 ({stage2['fail']/stage2['total']*100:.1f}%)")

    if stage2['offset_dist']:
        print(f"\n  [실제 offset 분포 (±7 범위 내)]")
        for offset in sorted(stage2['offset_dist'].keys()):
            count = stage2['offset_dist'][offset]
            pct = count / stage2['total'] * 100
            bar = '█' * int(pct / 2)
            print(f"    {offset:+2d}: {count:3d}회 ({pct:5.1f}%) {bar}")

    # 범위 밖 offset 분포
    out_of_range = {k: v for k, v in stage2['offset_all'].items() if k not in OFFSET_RANGE}
    if out_of_range:
        print(f"\n  [범위 밖 offset (±7 초과)]")
        for offset in sorted(out_of_range.keys()):
            count = out_of_range[offset]
            print(f"    {offset:+2d}: {count:3d}회")

    # 3단계: ord235
    print(f"\n{'=' * 70}")
    print(f"[3단계: ord235] - 점수 기반 Top-{TOP_N} 선택")
    print(f"{'=' * 70}")
    print(f"  ord2 통과: {stage3['ord2_pass']:4d}회 ({stage3['ord2_pass']/stage3['total']*100:.1f}%)")
    print(f"  ord3 통과: {stage3['ord3_pass']:4d}회 ({stage3['ord3_pass']/stage3['total']*100:.1f}%)")
    print(f"  ord5 통과: {stage3['ord5_pass']:4d}회 ({stage3['ord5_pass']/stage3['total']*100:.1f}%)")
    print(f"  전체 통과: {stage3['all_pass']:4d}회 ({stage3['all_pass']/stage3['total']*100:.1f}%)")

    # 순위 분포
    for pos in ['ord2', 'ord3', 'ord5']:
        dist = stage3['rank_dist'][pos]
        if dist:
            print(f"\n  [{pos} 실제 순위 분포]")
            total_in_range = sum(dist.values())
            cumulative = 0
            for rank in sorted(dist.keys())[:10]:
                count = dist[rank]
                cumulative += count
                pct = count / total_in_range * 100 if total_in_range > 0 else 0
                cum_pct = cumulative / total_in_range * 100 if total_in_range > 0 else 0
                marker = " ← Top-5" if rank == 5 else ""
                print(f"    {rank:2d}위: {count:3d}회 ({pct:5.1f}%) 누적 {cum_pct:5.1f}%{marker}")
            if len(dist) > 10:
                rest = sum(dist[r] for r in dist if r > 10)
                print(f"    11위+: {rest:3d}회")

    # 종합 분석
    print(f"\n{'=' * 70}")
    print(f"[종합 분석] - 6개 일치 가능 회차")
    print(f"{'=' * 70}")
    print(f"  1단계 실패: {full['stage1_fail']:4d}회 ({full['stage1_fail']/full['total']*100:.1f}%)")
    print(f"  2단계 실패: {full['stage2_fail']:4d}회 ({full['stage2_fail']/full['total']*100:.1f}%)")
    print(f"  3단계 실패: {full['stage3_fail']:4d}회 ({full['stage3_fail']/full['total']*100:.1f}%)")
    print(f"  6개 일치 가능: {full['success']:4d}회 ({full['success']/full['total']*100:.1f}%)")

    if full['success_rounds']:
        print(f"\n  [6개 일치 가능 회차] (최대 10개)")
        print(f"    {full['success_rounds'][:10]}")

    print(f"\n  [포지션별 실패 원인]")
    for pos, count in full['fail_by_position'].most_common():
        pct = count / full['total'] * 100
        bar = '█' * int(pct / 2)
        print(f"    {pos:15s}: {count:4d}회 ({pct:5.1f}%) {bar}")

    if full['5match_miss_position']:
        print(f"\n  [5개 적중 시 놓친 포지션]")
        for pos, count in full['5match_miss_position'].most_common():
            print(f"    {pos}: {count}회")

    # Top-N 비교
    print(f"\n{'=' * 70}")
    print(f"[Top-N 비교 분석]")
    print(f"{'=' * 70}")
    print(f"  {'Top-N':>6s} | {'ord2':>6s} | {'ord3':>6s} | {'ord5':>6s} | {'전체':>6s} | {'6개일치':>7s} | {'조합수':>12s}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*12}")
    for top_n, data in sorted(top_n_compare.items()):
        total = data['total']
        ord2_pct = data['ord2_pass'] / total * 100
        ord3_pct = data['ord3_pass'] / total * 100
        ord5_pct = data['ord5_pass'] / total * 100
        all_pct = data['all_pass'] / total * 100
        success_pct = data['success'] / total * 100
        marker = " ←현재" if top_n == TOP_N else ""
        print(f"  {top_n:>6d} | {ord2_pct:>5.1f}% | {ord3_pct:>5.1f}% | {ord5_pct:>5.1f}% | {all_pct:>5.1f}% | {success_pct:>6.1f}% | {data['combinations']:>12s}{marker}")

    # 개선 제안
    print(f"\n{'=' * 70}")
    print(f"[개선 제안]")
    print(f"{'=' * 70}")

    # 가장 큰 병목 찾기
    bottleneck = full['fail_by_position'].most_common(1)
    if bottleneck:
        main_cause, main_count = bottleneck[0]
        pct = main_count / full['total'] * 100
        print(f"\n  주요 병목: {main_cause} ({main_count}회, {pct:.1f}%)")

    # 개선안
    print(f"\n  [권장 개선안]")

    # ord2가 병목인 경우
    ord2_fail = full['fail_by_position'].get('ord2', 0)
    if ord2_fail > 0:
        # Top-7 적용 시 예상 통과율
        top7_ord2 = sum(stage3['rank_dist']['ord2'].get(r, 0) for r in range(1, 8))
        top7_total = sum(stage3['rank_dist']['ord2'].values())
        if top7_total > 0:
            top7_pct = top7_ord2 / top7_total * 100
            print(f"    1. ord2 top_n 확대: {TOP_N} → 7 (예상 {top7_pct:.1f}%)")

    # ord4가 병목인 경우
    ord4_fail = full['fail_by_position'].get('ord4 범위', 0)
    if ord4_fail > 0:
        # ±10 적용 시 예상 통과율
        pass_10 = sum(stage2['offset_all'].get(o, 0) for o in range(-10, 11))
        total = sum(stage2['offset_all'].values())
        if total > 0:
            pct_10 = pass_10 / total * 100
            print(f"    2. offset 범위 확대: ±7 → ±10 (예상 {pct_10:.1f}%)")

    # 점수화 개선
    print(f"\n  [점수화 개선 아이디어]")
    print(f"    - 최근 출현 번호 페널티 조정 (-5 → -3)")
    print(f"    - 연속 번호 패턴 반영 (1쌍 보너스)")
    print(f"    - 합계 범위 (121-160) 보너스 추가")


def main():
    parser = argparse.ArgumentParser(description='6개 일치 못하는 이유 분석')
    parser.add_argument('--start', type=int, default=826, help='시작 회차')
    parser.add_argument('--end', type=int, default=1204, help='종료 회차')
    args = parser.parse_args()

    all_data = load_all_data()

    # 범위 조정
    start_round = max(args.start, 826)
    end_round = min(args.end, all_data[-1]['round'])

    print(f"데이터 로드: {len(all_data)}회차")
    print(f"분석 범위: {start_round} ~ {end_round}")

    # 각 단계 분석
    stage1 = analyze_stage1_firstend(all_data, start_round, end_round)
    stage2 = analyze_stage2_ord4(all_data, start_round, end_round)
    stage3 = analyze_stage3_ord235(all_data, start_round, end_round, TOP_N)
    full = analyze_full_pipeline(all_data, start_round, end_round, TOP_N)

    # Top-N 비교
    top_n_compare = compare_top_n(all_data, start_round, end_round)

    # 리포트 출력
    print_report(stage1, stage2, stage3, full, top_n_compare, start_round, end_round)


if __name__ == "__main__":
    main()
