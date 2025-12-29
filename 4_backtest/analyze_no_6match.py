"""
6개 적중이 없는 이유 분석

각 단계에서 실제 당첨번호가 후보에서 탈락하는지 추적
"""

import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"

# 백테스트 파이프라인에서 가져온 함수들
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31, 37, 33, 10, 2, 32}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7, 30, 15, 11, 40, 6}

def get_range(num):
    if num <= 9: return 0
    elif num <= 19: return 1
    elif num <= 29: return 2
    elif num <= 39: return 3
    else: return 4

def load_all_data():
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'round': int(row['round']),
                'balls': tuple(sorted([int(row[f'ball{i}']) for i in range(1, 7)])),
            })
    return sorted(results, key=lambda x: x['round'])


# ============ Second 후보 선택 (top 2만 반환) ============
BALL2_STATS = {7: 25, 16: 23, 11: 23, 6: 23, 9: 23, 12: 21, 14: 20, 18: 20, 13: 20, 10: 17, 4: 17, 8: 16, 15: 16, 5: 13, 21: 12, 17: 12, 19: 11, 3: 9, 26: 8, 20: 8, 23: 8, 24: 6, 25: 5, 2: 5}
BALL1_BALL2_STATS = {1: {'avg': 7.6, 'mode': 9}, 2: {'avg': 9.1, 'mode': 6}, 3: {'avg': 9.2, 'mode': 6}, 4: {'avg': 10.4, 'mode': 7}, 5: {'avg': 11.3, 'mode': 12}, 6: {'avg': 12.4, 'mode': 7}, 7: {'avg': 12.4, 'mode': 9}, 8: {'avg': 14.3, 'mode': 11}, 9: {'avg': 14.4, 'mode': 14}, 10: {'avg': 16.5, 'mode': 16}}
GAP12_STATS = {1: 47, 2: 38, 3: 40, 4: 50, 5: 32, 6: 33, 7: 18, 8: 26, 9: 13, 10: 15}
ORD2_BIT_FREQ = {9: 28, 10: 26, 5: 23, 17: 22, 6: 21, 7: 20, 11: 20, 12: 20, 15: 20, 19: 19, 13: 18, 14: 17, 18: 15, 20: 14}

def get_ord2_candidates(ord1, ord6, max_candidates=2):
    """ord2 후보 전체 점수 반환"""
    min_ord2 = ord1 + 1
    max_ord2 = ord6 - 4
    if min_ord2 > max_ord2:
        return []

    candidates = []
    for ord2 in range(min_ord2, max_ord2 + 1):
        score = 0
        if ord2 in BALL2_STATS:
            score += BALL2_STATS[ord2] * 2
        if ord1 in BALL1_BALL2_STATS:
            stats = BALL1_BALL2_STATS[ord1]
            if ord2 == stats['mode']:
                score += 15
            if abs(ord2 - stats['avg']) <= 2:
                score += 10
        gap = ord2 - ord1
        if gap in GAP12_STATS:
            score += GAP12_STATS[gap] // 2
        if 10 <= ord2 <= 19:
            score += 10
        if ord2 in ORD2_BIT_FREQ:
            score += ORD2_BIT_FREQ[ord2]
        if ord2 in HOT_BITS:
            score += 10
        elif ord2 in COLD_BITS:
            score -= 5
        candidates.append({'ord2': ord2, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates


BALL3_STATS = {17: 25, 18: 23, 19: 22, 12: 21, 14: 21, 20: 20, 16: 20, 13: 19, 21: 18, 22: 18, 15: 17, 11: 16, 23: 15, 24: 14, 10: 13, 25: 12, 9: 11, 26: 10}
GAP23_STATS = {1: 55, 2: 48, 3: 43, 4: 37, 5: 32, 6: 28, 7: 22, 8: 19, 9: 15, 10: 13}
ORD3_BIT_FREQ = {17: 23, 18: 22, 19: 21, 12: 20, 14: 20, 20: 19, 16: 19, 13: 18, 21: 17, 22: 17, 15: 16, 11: 15}

def get_ord3_candidates(ord1, ord2, ord6, max_candidates=2):
    min_ord3 = ord2 + 1
    max_ord3 = ord6 - 3
    if min_ord3 > max_ord3:
        return []

    candidates = []
    for ord3 in range(min_ord3, max_ord3 + 1):
        score = 0
        if ord3 in BALL3_STATS:
            score += BALL3_STATS[ord3] * 2
        gap = ord3 - ord2
        if gap in GAP23_STATS:
            score += GAP23_STATS[gap] // 2
        if 10 <= ord3 <= 19:
            score += 10
        if ord3 in ORD3_BIT_FREQ:
            score += ORD3_BIT_FREQ[ord3]
        if ord3 in HOT_BITS:
            score += 10
        elif ord3 in COLD_BITS:
            score -= 5
        candidates.append({'ord3': ord3, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates


BALL4_STATS = {30: 26, 26: 19, 29: 18, 35: 18, 27: 18, 23: 18, 33: 17, 28: 16, 22: 16, 32: 16, 31: 16, 21: 15, 24: 15, 25: 14, 19: 14, 20: 13}
GAP34_STATS = {1: 46, 2: 54, 3: 44, 4: 37, 5: 24, 6: 30, 7: 25, 8: 22, 9: 18, 10: 18}
ORD4_BIT_FREQ = {27: 23, 28: 22, 25: 20, 24: 19, 29: 19, 31: 18, 21: 17, 17: 17, 32: 17, 33: 17, 23: 16}
UNEXPECTED_NUMBERS = {14, 37, 15, 36, 13, 34, 17, 40, 39, 41, 11, 12, 43, 42, 8}

def get_ord4_candidates(ord1, ord2, ord3, ord6, max_candidates=2):
    min_ord4 = ord3 + 1
    max_ord4 = ord6 - 1
    if min_ord4 > max_ord4:
        return []

    candidates = []
    for ord4 in range(min_ord4, max_ord4 + 1):
        score = 0
        if ord4 in BALL4_STATS:
            score += BALL4_STATS[ord4]
        if ord4 in UNEXPECTED_NUMBERS:
            score += 15
        if 20 <= ord4 <= 29:
            score += 8
        elif 30 <= ord4 <= 39:
            score += 6
        gap = ord4 - ord3
        if gap in GAP34_STATS:
            score += GAP34_STATS[gap] // 4
        if ord4 in ORD4_BIT_FREQ:
            score += ORD4_BIT_FREQ[ord4] // 2
        if ord4 in COLD_BITS:
            score += 12
        elif ord4 in HOT_BITS:
            score -= 3
        candidates.append({'ord4': ord4, 'score': score})

    candidates.sort(key=lambda x: -x['score'])
    return candidates


BALL5_STATS = {33: 26, 34: 24, 38: 23, 37: 22, 39: 21, 31: 20, 36: 20, 35: 19, 32: 18, 28: 18, 40: 17, 29: 17, 30: 16, 27: 15, 42: 14, 26: 13, 41: 13}
GAP45_STATS = {1: 56, 2: 51, 3: 42, 4: 35, 5: 33, 6: 29, 7: 26, 8: 19, 9: 18, 10: 14}

def get_ord5_candidates(ord1, ord2, ord3, ord4, ord6, winning_sets, max_candidates=2):
    min_ord5 = ord4 + 1
    max_ord5 = ord6 - 1
    if min_ord5 > max_ord5:
        return []

    SUM_OPTIMAL_MIN, SUM_OPTIMAL_MAX = 121, 160
    SUM_GOOD_MIN, SUM_GOOD_MAX = 100, 170
    current_sum = ord1 + ord2 + ord3 + ord4 + ord6

    candidates = []
    for ord5 in range(min_ord5, max_ord5 + 1):
        combo = tuple(sorted([ord1, ord2, ord3, ord4, ord5, ord6]))

        # 5개/6개 일치 체크 (제외 대상)
        is_duplicate = False
        for winning in winning_sets:
            if len(set(combo) & set(winning)) >= 5:
                is_duplicate = True
                break

        score = 0
        total_sum = current_sum + ord5

        if SUM_OPTIMAL_MIN <= total_sum <= SUM_OPTIMAL_MAX:
            score += 50
            mid = (SUM_OPTIMAL_MIN + SUM_OPTIMAL_MAX) / 2
            score += max(0, 20 - int(abs(total_sum - mid) / 2))
        elif SUM_GOOD_MIN <= total_sum <= SUM_GOOD_MAX:
            score += 25
        else:
            score -= 30

        if ord5 in BALL5_STATS:
            score += BALL5_STATS[ord5]
        gap = ord5 - ord4
        if gap in GAP45_STATS:
            score += GAP45_STATS[gap] // 4
        if ord5 in HOT_BITS:
            score += 10
        elif ord5 in COLD_BITS:
            score -= 3

        candidates.append({
            'ord5': ord5,
            'score': score,
            'is_duplicate': is_duplicate,
            'sum': total_sum
        })

    candidates.sort(key=lambda x: -x['score'])
    return candidates


def analyze_round(data, target_round):
    """특정 회차에서 실제 당첨번호가 어느 단계에서 탈락하는지 분석"""

    # 해당 회차 실제 당첨번호
    actual_data = next((d for d in data if d['round'] == target_round), None)
    if not actual_data:
        return None

    actual = actual_data['balls']  # 정렬된 6개 번호
    a1, a2, a3, a4, a5, a6 = actual

    # 학습 데이터 (target_round 이전)
    train_data = [d for d in data if d['round'] < target_round]
    winning_sets = [d['balls'] for d in train_data]

    result = {
        'round': target_round,
        'actual': actual,
        'failures': []
    }

    # ========== 1단계: Firstend (ord1, ord6) ==========
    # 모든 가능한 쌍 중 (a1, a6) 존재 여부
    actual_ord1_max = max(r['balls'][0] for r in train_data) if train_data else 10

    # firstend는 모든 가능 쌍을 생성하므로 항상 포함됨 (조건: 1 <= o1 <= actual_ord1_max, o1+5 <= o6 <= 45)
    firstend_valid = (1 <= a1 <= actual_ord1_max) and (a1 + 5 <= a6 <= 45)
    if not firstend_valid:
        result['failures'].append({
            'stage': 'firstend',
            'reason': f'ord1={a1}이 학습데이터 최대 ord1({actual_ord1_max})보다 크거나, (ord1, ord6) 범위 벗어남'
        })
        return result

    # ========== 2단계: Second (ord2) ==========
    ord2_candidates = get_ord2_candidates(a1, a6, max_candidates=2)
    all_ord2 = [c['ord2'] for c in ord2_candidates]
    top2_ord2 = all_ord2[:2]

    if a2 not in top2_ord2:
        # 순위 확인
        all_ord2_full = get_ord2_candidates(a1, a6, max_candidates=100)
        ranks = {c['ord2']: i+1 for i, c in enumerate(all_ord2_full)}
        actual_rank = ranks.get(a2, -1)

        result['failures'].append({
            'stage': 'second',
            'reason': f'실제 ord2={a2}가 top2에 없음 (순위: {actual_rank}/{len(all_ord2_full)})',
            'top2': top2_ord2,
            'actual_score': next((c['score'] for c in all_ord2_full if c['ord2'] == a2), None)
        })
        return result

    # ========== 3단계: Third (ord3) ==========
    ord3_candidates = get_ord3_candidates(a1, a2, a6, max_candidates=2)
    all_ord3 = [c['ord3'] for c in ord3_candidates]
    top2_ord3 = all_ord3[:2]

    if a3 not in top2_ord3:
        all_ord3_full = get_ord3_candidates(a1, a2, a6, max_candidates=100)
        ranks = {c['ord3']: i+1 for i, c in enumerate(all_ord3_full)}
        actual_rank = ranks.get(a3, -1)

        result['failures'].append({
            'stage': 'third',
            'reason': f'실제 ord3={a3}가 top2에 없음 (순위: {actual_rank}/{len(all_ord3_full)})',
            'top2': top2_ord3,
            'actual_score': next((c['score'] for c in all_ord3_full if c['ord3'] == a3), None)
        })
        return result

    # ========== 4단계: Fourth (ord4) ==========
    ord4_candidates = get_ord4_candidates(a1, a2, a3, a6, max_candidates=2)
    all_ord4 = [c['ord4'] for c in ord4_candidates]
    top2_ord4 = all_ord4[:2]

    if a4 not in top2_ord4:
        all_ord4_full = get_ord4_candidates(a1, a2, a3, a6, max_candidates=100)
        ranks = {c['ord4']: i+1 for i, c in enumerate(all_ord4_full)}
        actual_rank = ranks.get(a4, -1)

        result['failures'].append({
            'stage': 'fourth',
            'reason': f'실제 ord4={a4}가 top2에 없음 (순위: {actual_rank}/{len(all_ord4_full)})',
            'top2': top2_ord4,
            'actual_score': next((c['score'] for c in all_ord4_full if c['ord4'] == a4), None)
        })
        return result

    # ========== 5단계: Fifth (ord5) ==========
    ord5_candidates = get_ord5_candidates(a1, a2, a3, a4, a6, winning_sets, max_candidates=2)
    all_ord5 = [c['ord5'] for c in ord5_candidates if not c['is_duplicate']]

    # 실제 ord5가 중복 제외 대상인지 확인
    all_ord5_full = get_ord5_candidates(a1, a2, a3, a4, a6, winning_sets, max_candidates=100)
    actual_ord5_info = next((c for c in all_ord5_full if c['ord5'] == a5), None)

    if actual_ord5_info and actual_ord5_info['is_duplicate']:
        result['failures'].append({
            'stage': 'fifth',
            'reason': f'실제 ord5={a5}가 5/6개 중복으로 제외됨',
            'is_duplicate': True
        })
        return result

    # 중복이 아닌 후보 중 top2
    non_dup_candidates = [c for c in all_ord5_full if not c['is_duplicate']]
    top2_ord5 = [c['ord5'] for c in non_dup_candidates[:2]]

    if a5 not in top2_ord5:
        ranks = {c['ord5']: i+1 for i, c in enumerate(non_dup_candidates)}
        actual_rank = ranks.get(a5, -1)

        result['failures'].append({
            'stage': 'fifth',
            'reason': f'실제 ord5={a5}가 top2에 없음 (순위: {actual_rank}/{len(non_dup_candidates)})',
            'top2': top2_ord5,
            'actual_score': actual_ord5_info['score'] if actual_ord5_info else None,
            'actual_sum': actual_ord5_info['sum'] if actual_ord5_info else None
        })
        return result

    # 모든 단계 통과! (6개 적중 가능)
    result['all_passed'] = True
    return result


def main():
    print("=" * 80)
    print("6개 적중 부재 원인 분석")
    print("=" * 80)

    data = load_all_data()

    # 분석 범위
    start_round, end_round = 900, 1100

    # 단계별 탈락 통계
    failure_stats = {
        'firstend': 0,
        'second': 0,
        'third': 0,
        'fourth': 0,
        'fifth': 0,
        'fifth_duplicate': 0
    }

    all_passed = 0
    total_rounds = 0

    # 상세 분석 결과
    detailed_failures = []

    for target_round in range(start_round, end_round + 1):
        result = analyze_round(data, target_round)
        if result is None:
            continue

        total_rounds += 1

        if result.get('all_passed'):
            all_passed += 1
            print(f"[{target_round}회차] ✓ 6개 적중 가능! 실제: {result['actual']}")
        else:
            failure = result['failures'][0]  # 첫 탈락 지점
            stage = failure['stage']
            if stage == 'fifth' and failure.get('is_duplicate'):
                failure_stats['fifth_duplicate'] += 1
            else:
                failure_stats[stage] += 1

            detailed_failures.append({
                'round': target_round,
                'actual': result['actual'],
                'stage': stage,
                'reason': failure['reason']
            })

    # 결과 요약
    print("\n" + "=" * 80)
    print(f"[결과 요약] 분석 회차: {total_rounds}회")
    print("=" * 80)

    print("\n<단계별 탈락 통계>")
    for stage, count in failure_stats.items():
        pct = count / total_rounds * 100 if total_rounds > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {stage:20s}: {count:4d}회 ({pct:5.1f}%) {bar}")

    print(f"\n  6개 적중 가능 회차: {all_passed}회 ({all_passed/total_rounds*100:.1f}%)")

    # 각 단계별 상세 분석
    print("\n" + "=" * 80)
    print("<단계별 탈락 원인 분석>")
    print("=" * 80)

    for stage in ['second', 'third', 'fourth', 'fifth']:
        stage_failures = [f for f in detailed_failures if f['stage'] == stage]
        if stage_failures:
            print(f"\n[{stage} 단계 탈락 샘플 (최근 5건)]")
            for f in stage_failures[-5:]:
                print(f"  {f['round']}회차: {f['actual']} - {f['reason']}")


if __name__ == "__main__":
    main()
