"""
왜 6개 일치를 못하는가? - 단계별 병목 분석

main.py 파이프라인:
1. FirstEnd: (ord1, ord6) 쌍 477개 생성
2. ord4: 공식 A + ±7 범위 (15개 offset)
3. ord235: 각 포지션 점수 최고 1개만 선택

이 스크립트는 각 단계에서 실제 당첨번호가 탈락하는 비율을 분석합니다.

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

# 상수 (main.py와 동일)
FORMULA_RATIO = 0.60
OFFSET_RANGE = range(-7, 8)

OPTIMAL_RANGES = {
    'ord2': (10, 19),
    'ord3': (10, 19),
    'ord5': (30, 39),
}

HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


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
    }

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        results['total'] += 1
        actual_ord1, actual_ord6 = winning[0], winning[5]

        # 477개 쌍 범위: ord1 <= 21, ord6 >= 23, 최소 5칸 차이
        # ord1: 1~21, ord6: max(ord1+5, 23)~45
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
        'offset_dist': Counter(),  # 실제 offset 분포
    }

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_numbers(all_data, target_round)
        if winning is None:
            continue

        results['total'] += 1
        ord1, ord4, ord6 = winning[0], winning[3], winning[5]

        # 공식 A 계산
        base_ord4 = round(ord1 + (ord6 - ord1) * FORMULA_RATIO)
        actual_offset = ord4 - base_ord4

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
# 3단계 분석: ord235 - 점수 기반 1개 선택
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


def get_best_candidate(candidates, pos_name, pos_freq, all_freq, recent_3):
    """범위 내에서 최고 점수 후보 선택"""
    if not candidates:
        return None, -999

    best_num = None
    best_score = -999

    for num in candidates:
        score = score_candidate(num, pos_name, pos_freq, all_freq, recent_3)
        if score > best_score:
            best_score = score
            best_num = num

    return best_num, best_score


def analyze_stage3_ord235(all_data, start_round, end_round):
    """3단계: ord2, ord3, ord5가 점수 최고인지 분석"""
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

        # ord2 분석
        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        best_ord2, best_score2 = get_best_candidate(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3)

        if ord2 in ord2_candidates:
            # 실제 ord2의 순위 계산
            scores = [(n, score_candidate(n, 'ord2', pos_freq, all_freq, recent_3)) for n in ord2_candidates]
            scores.sort(key=lambda x: -x[1])
            rank = next(i for i, (n, _) in enumerate(scores, 1) if n == ord2)
            results['rank_dist']['ord2'][rank] += 1

            if ord2 == best_ord2:
                results['ord2_pass'] += 1
            else:
                results['ord2_fail_details'].append({
                    'round': target_round,
                    'actual': ord2,
                    'predicted': best_ord2,
                    'rank': rank,
                    'candidates': len(ord2_candidates),
                })
        else:
            # 후보 범위 밖
            results['ord2_fail_details'].append({
                'round': target_round,
                'actual': ord2,
                'predicted': best_ord2,
                'rank': 'N/A',
                'reason': 'out of range',
            })

        # ord3 분석 (실제 ord2 기준)
        ord3_candidates = list(range(ord2 + 1, ord4))
        best_ord3, best_score3 = get_best_candidate(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3)

        if ord3 in ord3_candidates:
            scores = [(n, score_candidate(n, 'ord3', pos_freq, all_freq, recent_3)) for n in ord3_candidates]
            scores.sort(key=lambda x: -x[1])
            rank = next(i for i, (n, _) in enumerate(scores, 1) if n == ord3)
            results['rank_dist']['ord3'][rank] += 1

            if ord3 == best_ord3:
                results['ord3_pass'] += 1
            else:
                results['ord3_fail_details'].append({
                    'round': target_round,
                    'actual': ord3,
                    'predicted': best_ord3,
                    'rank': rank,
                })
        else:
            results['ord3_fail_details'].append({
                'round': target_round,
                'actual': ord3,
                'predicted': best_ord3,
                'rank': 'N/A',
                'reason': 'out of range',
            })

        # ord5 분석
        ord5_candidates = list(range(ord4 + 1, ord6))
        best_ord5, best_score5 = get_best_candidate(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3)

        if ord5 in ord5_candidates:
            scores = [(n, score_candidate(n, 'ord5', pos_freq, all_freq, recent_3)) for n in ord5_candidates]
            scores.sort(key=lambda x: -x[1])
            rank = next(i for i, (n, _) in enumerate(scores, 1) if n == ord5)
            results['rank_dist']['ord5'][rank] += 1

            if ord5 == best_ord5:
                results['ord5_pass'] += 1
            else:
                results['ord5_fail_details'].append({
                    'round': target_round,
                    'actual': ord5,
                    'predicted': best_ord5,
                    'rank': rank,
                })
        else:
            results['ord5_fail_details'].append({
                'round': target_round,
                'actual': ord5,
                'predicted': best_ord5,
                'rank': 'N/A',
                'reason': 'out of range',
            })

        # 모두 통과?
        if (ord2 == best_ord2 and ord3 == best_ord3 and ord5 == best_ord5):
            results['all_pass'] += 1

    return results


# ============================================================
# 종합 분석: 6개 일치 가능 회차 분석
# ============================================================

def analyze_full_pipeline(all_data, start_round, end_round):
    """전체 파이프라인 시뮬레이션 - 6개 일치 가능 여부"""
    results = {
        'total': 0,
        'stage1_fail': 0,
        'stage2_fail': 0,
        'stage3_fail': 0,
        'success': 0,  # 6개 일치 가능한 회차
        'success_rounds': [],
        'fail_by_position': Counter(),  # 어느 포지션에서 실패했는지
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

        # 3단계: ord235 체크
        pos_freq, all_freq, recent_3 = get_position_stats(train_data)

        fail_positions = []

        # ord2 (실제 ord1, ord4 기준 범위)
        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        best_ord2, _ = get_best_candidate(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3)
        if ord2 != best_ord2:
            fail_positions.append('ord2')

        # ord3 (실제 ord2, ord4 기준 범위)
        ord3_candidates = list(range(ord2 + 1, ord4))
        best_ord3, _ = get_best_candidate(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3)
        if ord3 != best_ord3:
            fail_positions.append('ord3')

        # ord5 (실제 ord4, ord6 기준 범위)
        ord5_candidates = list(range(ord4 + 1, ord6))
        best_ord5, _ = get_best_candidate(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3)
        if ord5 != best_ord5:
            fail_positions.append('ord5')

        if fail_positions:
            results['stage3_fail'] += 1
            for pos in fail_positions:
                results['fail_by_position'][pos] += 1
        else:
            results['success'] += 1
            results['success_rounds'].append(target_round)

    return results


def print_report(stage1, stage2, stage3, full, start_round, end_round):
    """분석 결과 출력"""
    print("=" * 70)
    print(f"왜 6개 일치를 못하는가? - 병목 분석")
    print(f"분석 범위: {start_round} ~ {end_round}회차 ({stage1['total']}개)")
    print("=" * 70)

    # 1단계: FirstEnd
    print(f"\n{'=' * 70}")
    print(f"[1단계: FirstEnd] - (ord1, ord6) 쌍 생성")
    print(f"{'=' * 70}")
    print(f"  통과: {stage1['pass']:4d}회 ({stage1['pass']/stage1['total']*100:.1f}%)")
    print(f"  실패: {stage1['fail']:4d}회 ({stage1['fail']/stage1['total']*100:.1f}%)")
    print(f"  → 범위: ord1 1~21, ord6 23~45, 최소 5칸 차이")
    if stage1['fail'] > 0:
        print(f"\n  [실패 사례 (최대 5개)]")
        for detail in stage1['fail_details'][:5]:
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
            bar = '█' * (count // 5)
            print(f"    {offset:+2d}: {count:3d}회 {bar}")

    if stage2['fail'] > 0:
        print(f"\n  [실패 사례 (최대 5개)]")
        for detail in stage2['fail_details'][:5]:
            print(f"    {detail['round']}회: ord4={detail['ord4']}, base={detail['base_ord4']}, offset={detail['actual_offset']:+d}")

    # 3단계: ord235
    print(f"\n{'=' * 70}")
    print(f"[3단계: ord235] - 점수 기반 최고 1개 선택")
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
            for rank in sorted(dist.keys())[:5]:
                count = dist[rank]
                pct = count / total_in_range * 100 if total_in_range > 0 else 0
                bar = '█' * int(pct / 5)
                print(f"    {rank}위: {count:3d}회 ({pct:5.1f}%) {bar}")
            if len(dist) > 5:
                rest = sum(dist[r] for r in dist if r > 5)
                pct = rest / total_in_range * 100 if total_in_range > 0 else 0
                print(f"    6위+: {rest:3d}회 ({pct:5.1f}%)")

    # 종합 분석
    print(f"\n{'=' * 70}")
    print(f"[종합 분석] - 6개 일치 가능 회차")
    print(f"{'=' * 70}")
    print(f"  1단계 실패: {full['stage1_fail']:4d}회 ({full['stage1_fail']/full['total']*100:.1f}%)")
    print(f"  2단계 실패: {full['stage2_fail']:4d}회 ({full['stage2_fail']/full['total']*100:.1f}%)")
    print(f"  3단계 실패: {full['stage3_fail']:4d}회 ({full['stage3_fail']/full['total']*100:.1f}%)")
    print(f"  6개 일치 가능: {full['success']:4d}회 ({full['success']/full['total']*100:.1f}%)")

    if full['success_rounds']:
        print(f"\n  [6개 일치 가능 회차]")
        print(f"    {full['success_rounds']}")

    print(f"\n  [포지션별 실패 원인]")
    for pos, count in full['fail_by_position'].most_common():
        pct = count / full['total'] * 100
        bar = '█' * int(pct / 2)
        print(f"    {pos:15s}: {count:4d}회 ({pct:5.1f}%) {bar}")

    # 결론
    print(f"\n{'=' * 70}")
    print(f"[결론] - 왜 6개 일치가 어려운가?")
    print(f"{'=' * 70}")

    # 가장 큰 병목 찾기
    bottleneck = full['fail_by_position'].most_common(1)
    if bottleneck:
        main_cause, main_count = bottleneck[0]
        print(f"\n  주요 원인: {main_cause} ({main_count}회, {main_count/full['total']*100:.1f}%)")

    ord235_fail = full['fail_by_position'].get('ord2', 0) + full['fail_by_position'].get('ord3', 0) + full['fail_by_position'].get('ord5', 0)
    if ord235_fail > 0:
        print(f"\n  3단계(ord235) 문제:")
        print(f"    - 현재: 각 포지션에서 점수 최고 1개만 선택")
        print(f"    - 문제: 실제 당첨번호가 1위가 아닌 경우가 많음")

        # ord2의 2~5위 비율
        ord2_not_1st = sum(stage3['rank_dist']['ord2'].get(r, 0) for r in range(2, 6))
        ord2_total = sum(stage3['rank_dist']['ord2'].values())
        if ord2_total > 0:
            print(f"    - ord2: 2~5위에 {ord2_not_1st}회 ({ord2_not_1st/ord2_total*100:.1f}%)")

    stage12_fail = full['stage1_fail'] + full['stage2_fail']
    if stage12_fail > 0:
        print(f"\n  1~2단계(firstend + ord4) 문제:")
        print(f"    - 실패: {stage12_fail}회 ({stage12_fail/full['total']*100:.1f}%)")
        print(f"    - 해결책: 범위 확장 필요 (현재 ±7)")

    print(f"\n  개선 제안:")
    print(f"    1. 3단계: Top-N 후보 선택 (1개 → 3~5개)")
    print(f"    2. 2단계: offset 범위 확장 (±7 → ±10)")
    print(f"    3. 점수화: 순위 기반 가중치 조정")

    # 예상 개선 효과
    print(f"\n  예상 개선 효과:")
    # Top-3 적용 시 ord2 통과율
    ord2_top3 = sum(stage3['rank_dist']['ord2'].get(r, 0) for r in range(1, 4))
    ord2_total = sum(stage3['rank_dist']['ord2'].values())
    if ord2_total > 0:
        print(f"    - ord2 Top-3: {ord2_top3/ord2_total*100:.1f}% (현재 {stage3['ord2_pass']/stage3['total']*100:.1f}%)")


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
    stage3 = analyze_stage3_ord235(all_data, start_round, end_round)
    full = analyze_full_pipeline(all_data, start_round, end_round)

    # 리포트 출력
    print_report(stage1, stage2, stage3, full, start_round, end_round)


if __name__ == "__main__":
    main()
