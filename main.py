"""
로또 예측 시스템 - 메인 실행 파일

전체 파이프라인:
1. 데이터 로드 (1_data)
2. 인사이트 학습 (2_insights) - 목표 회차 직전까지
3. firstend: (ord1, ord6) 쌍 생성 (3_predict/1_firstend)
4. ord4 계산 (3_predict/2_146)
5. ord2, ord3, ord5 채우기 (3_predict/3_235)
6. 결과 저장 (result/)

실행:
    python main.py --round 1205
"""

import csv
import subprocess
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "1_data" / "winning_numbers.csv"
INSIGHTS_DIR = BASE_DIR / "2_insights"
RESULT_DIR = BASE_DIR / "result"


# ============================================================
# 상수 정의
# ============================================================

# ord4 공식: A (0.60) + ±10 범위
FORMULA_RATIO = 0.60
OFFSET_RANGE = range(-10, 11)  # -10 ~ +10 (21개)


# ============================================================
# 인사이트 로드 (동적)
# ============================================================

def load_insights():
    """인사이트 통계 파일에서 동적으로 로드"""
    insights = {
        'hot_bits': set(),
        'cold_bits': set(),
        'primes': set(),
        'optimal_ranges': {},
    }

    # 1. HOT/COLD bits 로드
    hot_cold_path = INSIGHTS_DIR / "7_onehot" / "statistics" / "hot_cold_bits.csv"
    if hot_cold_path.exists():
        with open(hot_cold_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['category'] == 'HOT':
                    insights['hot_bits'].add(int(row['bit']))
                elif row['category'] == 'COLD':
                    insights['cold_bits'].add(int(row['bit']))
        print(f"    - 7_onehot: HOT {len(insights['hot_bits'])}개, COLD {len(insights['cold_bits'])}개")
    else:
        print(f"    - 7_onehot: 파일 없음 (기본값 사용)")
        insights['hot_bits'] = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
        insights['cold_bits'] = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

    # 2. 소수 로드
    prime_path = INSIGHTS_DIR / "5_prime" / "statistics" / "prime_frequency.csv"
    if prime_path.exists():
        with open(prime_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                insights['primes'].add(int(row['prime']))
        print(f"    - 5_prime: {len(insights['primes'])}개 소수")
    else:
        print(f"    - 5_prime: 파일 없음 (기본값 사용)")
        insights['primes'] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

    # 3. 최빈 구간 로드 (ord2, ord3, ord5, ord6)
    range_path = INSIGHTS_DIR / "4_range" / "statistics" / "position_range_distribution.csv"
    if range_path.exists():
        best_range = {}
        with open(range_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos = int(row['position'])
                prob = float(row['probability'])
                # ord2, ord3, ord5, ord6 모두 포함
                if pos in [2, 3, 5, 6]:
                    if pos not in best_range or prob > best_range[pos][1]:
                        range_idx = int(row['range'])
                        best_range[pos] = (range_idx, prob)

        # range_idx → (min, max) 변환
        range_map = {0: (1, 9), 1: (10, 19), 2: (20, 29), 3: (30, 39), 4: (40, 45)}
        for pos, (range_idx, _) in best_range.items():
            insights['optimal_ranges'][f'ord{pos}'] = range_map[range_idx]

        # ord3 범위 조정: 실제 평균 19.4이므로 (15, 24)가 더 적합
        if 'ord3' in insights['optimal_ranges']:
            insights['optimal_ranges']['ord3'] = (15, 24)

        print(f"    - 4_range: {insights['optimal_ranges']}")
    else:
        print(f"    - 4_range: 파일 없음 (기본값 사용)")
        # 8개 인사이트 분석 기반 최적 범위
        insights['optimal_ranges'] = {
            'ord2': (10, 19),   # 48.3%
            'ord3': (15, 24),   # 평균 19.4 기반 조정
            'ord5': (30, 39),   # 54.6%
            'ord6': (40, 45),   # 57.8%
        }

    return insights


def filter_6match_pattern(combo, insights):
    """6개 일치 가능성 높은 조합만 필터링

    기준: prime_count >= 2 (89% 유지, 39% 감소)
    - 6개 일치 조합: 평균 소수 2.78개
    - 5개 일치 조합: 평균 소수 1.92개
    """
    numbers = [combo['ord1'], combo['ord2'], combo['ord3'],
               combo['ord4'], combo['ord5'], combo['ord6']]

    # 소수 개수 계산 (insights에서 로드)
    primes = insights.get('primes', {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43})
    prime_count = sum(1 for n in numbers if n in primes)

    return prime_count >= 2


# ============================================================
# 1. 데이터 로드
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
# 2. 인사이트 학습 (2_insights 하위 폴더 전체 실행)
# ============================================================

def run_insights(target_round, all_data):
    """모든 인사이트 생성 스크립트 실행"""
    print(f"\n[인사이트 학습] 목표 회차: {target_round}")
    print(f"  학습 데이터: {target_round - 1}회차까지 ({target_round - 826}개)")

    # 인사이트 폴더 목록
    insight_folders = sorted([
        d for d in INSIGHTS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    print(f"  발견된 인사이트: {len(insight_folders)}개")

    for folder in insight_folders:
        generate_script = folder / "generate.py"
        if generate_script.exists():
            print(f"    - {folder.name}: ", end="")
            try:
                # 인사이트 생성 실행 (출력 억제)
                result = subprocess.run(
                    [sys.executable, str(generate_script)],
                    capture_output=True,
                    text=True,
                    cwd=str(folder),
                    timeout=60
                )
                if result.returncode == 0:
                    print("완료")
                else:
                    print(f"실패 ({result.stderr[:50]}...)")
            except subprocess.TimeoutExpired:
                print("타임아웃")
            except Exception as e:
                print(f"오류: {e}")
        else:
            print(f"    - {folder.name}: generate.py 없음")


# ============================================================
# 3. FirstEnd: (ord1, ord6) 쌍 생성
# ============================================================

def generate_firstend_pairs(data):
    """(ord1, ord6) 쌍 빈도 계산 - 전체 477개 쌍"""
    pair_rounds = defaultdict(list)

    for r in data:
        balls = r['balls']
        ord1, ord6 = balls[0], balls[5]
        pair_rounds[(ord1, ord6)].append(r['round'])

    # 모든 가능한 쌍 생성 (ord1 <= 21, ord6 >= 23, 최소 5칸 차이)
    all_pairs = []
    for ord1 in range(1, 22):  # 1~21
        for ord6 in range(max(ord1 + 5, 23), 46):  # ord1+5 ~ 45, ord6 >= 23
            rounds = pair_rounds.get((ord1, ord6), [])
            freq = len(rounds)
            all_pairs.append({
                'ord1': ord1,
                'ord6': ord6,
                'freq': freq,
                'rounds': rounds,
            })

    # 빈도순 정렬 (높은 순)
    all_pairs.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord6']))

    print(f"\n[1단계: FirstEnd] (ord1, ord6) 쌍 생성")
    print(f"  전체 쌍: {len(all_pairs)}개")
    print(f"  출현 쌍: {sum(1 for p in all_pairs if p['freq'] > 0)}개")
    print(f"  미출현 쌍: {sum(1 for p in all_pairs if p['freq'] == 0)}개")

    return all_pairs


# ============================================================
# 4. ord4 계산 (공식 A + ±7 범위)
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

    print(f"\n[2단계: ord4] 공식 A + ±10 범위")
    print(f"  생성된 행: {len(rows)}개")

    return rows


# ============================================================
# 5. ord2, ord3, ord5 채우기
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


def score_candidate(num, pos_name, pos_freq, all_freq, recent_3, insights):
    """후보 번호 점수 계산 - 8개 인사이트 활용, 포지션별 차별화

    [적용 인사이트]
    - 4_range: 포지션별 최빈 구간 보너스
    - 5_prime: 소수 보너스 (ord5 제외 - ball5 소수 18.5%로 낮음)
    - 7_onehot: HOT/COLD 비트 (ord5는 HOT +8, COLD -5로 강화)
    """
    seen_numbers = set(pos_freq[pos_name].keys())
    optimal_min, optimal_max = insights['optimal_ranges'].get(pos_name, (1, 45))

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
        if num in insights['hot_bits']:
            score += 8  # 기존 5 → 8 (강화)
        if num in insights['cold_bits']:
            score -= 5  # 기존 3 → 5 (강화)
    else:
        if num in insights['hot_bits']:
            score += 5
        if num in insights['cold_bits']:
            score -= 3

    # 최근 3회 출현 페널티
    if num in recent_3:
        score -= 5

    # [5. prime] 소수 보너스 - ord5 제외
    # ball5 소수 비율이 18.5%로 낮음 → ord5는 소수 보너스 제거
    if pos_name != 'ord5':
        if num in insights['primes']:
            score += 3

    return score


def find_top_candidates(candidates, pos_name, pos_freq, all_freq, recent_3, insights, top_n=15):
    """범위 내에서 상위 N개 후보 선택"""
    if not candidates:
        return []

    scored = []
    for num in candidates:
        score = score_candidate(num, pos_name, pos_freq, all_freq, recent_3, insights)
        scored.append((num, score))

    # 점수순 정렬
    scored.sort(key=lambda x: -x[1])

    # 상위 N개 반환
    return [num for num, _ in scored[:top_n]]


def fill_ord235(rows, pos_freq, all_freq, recent_3, insights, top_n=5):
    """ord2, ord3, ord5 채우기 - 각 포지션 상위 N개 조합"""
    new_rows = []

    for row in rows:
        ord1 = row['ord1']
        ord4 = row['ord4']
        ord6 = row['ord6']

        # ord2 후보: ord1+1 ~ ord4-2, 상위 N개
        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        ord2_picks = find_top_candidates(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3, insights, top_n)
        if not ord2_picks:
            continue

        # ord5 후보: ord4+1 ~ ord6-1, 상위 N개
        ord5_candidates = list(range(ord4 + 1, ord6))
        ord5_picks = find_top_candidates(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3, insights, top_n)
        if not ord5_picks:
            continue

        # 각 ord2에 대해 ord3 후보 생성
        for ord2 in ord2_picks:
            # ord3 후보: ord2+1 ~ ord4-1, 상위 N개
            ord3_candidates = list(range(ord2 + 1, ord4))
            ord3_picks = find_top_candidates(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3, insights, top_n)
            if not ord3_picks:
                continue

            for ord3 in ord3_picks:
                for ord5 in ord5_picks:
                    # 유효성 검사
                    if not (ord1 < ord2 < ord3 < ord4 < ord5 < ord6):
                        continue

                    combo = {
                        'ord1': ord1,
                        'ord2': ord2,
                        'ord3': ord3,
                        'ord4': ord4,
                        'ord5': ord5,
                        'ord6': ord6,
                        'freq': row['freq'],
                        'rounds': row['rounds'],
                        'offset': row['offset'],
                    }

                    # why.py와 동일: 소수 필터 제거
                    new_rows.append(combo)

    # 빈도순 정렬
    new_rows.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord4'], x['ord6']))

    print(f"\n[3단계: ord235] ord2, ord3, ord5 채우기 (Top-{top_n})")
    print(f"  생성된 조합: {len(new_rows)}개")

    return new_rows


# ============================================================
# 6. 결과 저장
# ============================================================

def save_result(rows, target_round):
    """result.csv 저장"""
    RESULT_DIR.mkdir(exist_ok=True)
    result_path = RESULT_DIR / "result.csv"

    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6', '빈도수', '회차', 'offset'])

        for row in rows:
            rounds_str = ','.join(map(str, row['rounds'])) if row['rounds'] else ''
            writer.writerow([
                row['ord1'],
                row['ord2'],
                row['ord3'],
                row['ord4'],
                row['ord5'],
                row['ord6'],
                row['freq'],
                rounds_str,
                row['offset'],
            ])

    print(f"\n[결과 저장] {result_path}")
    print(f"  총 {len(rows)}개 조합")

    return result_path


def check_winning_match(predictions, winning, target_round):
    """당첨번호와 예측 결과 일치 여부 확인"""
    print("\n" + "=" * 60)
    print(f"[{target_round}회차 당첨번호 검증]")
    print("=" * 60)

    if winning is None:
        print(f"  ⚠️  {target_round}회차 당첨번호가 데이터에 없습니다.")
        print(f"     (아직 추첨 전이거나 데이터 미갱신)")
        return None

    winning_set = set(winning)
    print(f"\n  당첨번호: {winning}")

    # 각 예측 조합과 비교
    best_match = 0
    best_prediction = None
    best_rank = 0
    match_counts = Counter()

    for rank, pred in enumerate(predictions, 1):
        pred_set = {pred['ord1'], pred['ord2'], pred['ord3'], pred['ord4'], pred['ord5'], pred['ord6']}
        match_count = len(pred_set & winning_set)
        match_counts[match_count] += 1

        if match_count > best_match:
            best_match = match_count
            best_prediction = pred
            best_rank = rank

    # 매치 분포 출력
    print(f"\n  [매치 분포]")
    for i in range(7):
        count = match_counts.get(i, 0)
        pct = count / len(predictions) * 100 if predictions else 0
        bar = '█' * int(pct / 5) if pct > 0 else ''
        print(f"    {i}개 적중: {count:5d}개 ({pct:5.1f}%) {bar}")

    # 최고 매치 결과
    print(f"\n  [최고 적중]")
    if best_prediction:
        pred_list = [best_prediction['ord1'], best_prediction['ord2'], best_prediction['ord3'],
                     best_prediction['ord4'], best_prediction['ord5'], best_prediction['ord6']]
        matched = sorted(set(pred_list) & winning_set)
        print(f"    적중 개수: {best_match}개")
        print(f"    순위: {best_rank}위 / {len(predictions)}개")
        print(f"    예측: {pred_list}")
        print(f"    매치: {matched}")
        print(f"    빈도: {best_prediction['freq']}, offset: {best_prediction['offset']:+d}")

    # 적중률 요약
    match_3plus = sum(match_counts.get(i, 0) for i in range(3, 7))
    match_4plus = sum(match_counts.get(i, 0) for i in range(4, 7))
    print(f"\n  [적중률 요약]")
    print(f"    3개+ 적중: {match_3plus}개 ({match_3plus/len(predictions)*100:.1f}%)")
    print(f"    4개+ 적중: {match_4plus}개 ({match_4plus/len(predictions)*100:.1f}%)")

    return {
        'round': target_round,
        'winning': winning,
        'best_match': best_match,
        'best_rank': best_rank,
        'best_prediction': best_prediction,
        'total_combinations': len(predictions),
        'match_counts': dict(match_counts),
    }


def save_match_distribution(results_list):
    """매치 분포를 result/match_distribution.csv에 저장"""
    RESULT_DIR.mkdir(exist_ok=True)
    result_path = RESULT_DIR / "match_distribution.csv"

    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'total_combinations',
                         'match_0', 'match_1', 'match_2', 'match_3', 'match_4', 'match_5', 'match_6',
                         'pct_0', 'pct_1', 'pct_2', 'pct_3', 'pct_4', 'pct_5', 'pct_6',
                         'best_match', 'best_rank', 'best_prediction'])

        for r in results_list:
            match_counts = r['match_counts']
            total = r['total_combinations']

            # 매치 개수 (0~6)
            counts = [match_counts.get(i, 0) for i in range(7)]
            # 퍼센트 (0~6)
            pcts = [counts[i] / total * 100 if total > 0 else 0 for i in range(7)]

            # best_prediction 문자열
            bp = r['best_prediction']
            if bp:
                bp_str = f"[{bp['ord1']},{bp['ord2']},{bp['ord3']},{bp['ord4']},{bp['ord5']},{bp['ord6']}]"
            else:
                bp_str = "[]"

            writer.writerow([
                r['round'],
                str(r['winning']),
                total,
                *counts,
                *[f"{p:.1f}" for p in pcts],
                r['best_match'],
                r['best_rank'],
                bp_str,
            ])

    print(f"\n[매치 분포 저장] {result_path}")
    return result_path


def print_summary(rows, target_round):
    """결과 요약 출력"""
    print("\n" + "=" * 60)
    print(f"예측 완료: {target_round}회차")
    print("=" * 60)

    print(f"\n총 조합 수: {len(rows)}개")

    # 빈도별 분포
    freq_dist = Counter(row['freq'] for row in rows)
    print(f"\n[빈도별 조합 분포]")
    for freq in sorted(freq_dist.keys(), reverse=True)[:10]:
        print(f"  빈도 {freq}: {freq_dist[freq]}개")

    # 상위 10개 조합
    print(f"\n[상위 10개 조합]")
    for i, row in enumerate(rows[:10], 1):
        combo = f"({row['ord1']}, {row['ord2']}, {row['ord3']}, {row['ord4']}, {row['ord5']}, {row['ord6']})"
        print(f"  {i:2d}. {combo} - 빈도 {row['freq']}, offset {row['offset']:+d}")

    print("\n" + "=" * 60)


# ============================================================
# 백테스트 모드
# ============================================================

def load_insights_silent():
    """인사이트 로드 (출력 없이)"""
    insights = {
        'hot_bits': set(),
        'cold_bits': set(),
        'primes': set(),
        'optimal_ranges': {},
    }

    # 1. HOT/COLD bits 로드
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

    # 2. 소수 로드
    prime_path = INSIGHTS_DIR / "5_prime" / "statistics" / "prime_frequency.csv"
    if prime_path.exists():
        with open(prime_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                insights['primes'].add(int(row['prime']))
    else:
        insights['primes'] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

    # 3. 최빈 구간 로드 (ord2, ord3, ord5, ord6)
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
        # ord3 범위 조정
        if 'ord3' in insights['optimal_ranges']:
            insights['optimal_ranges']['ord3'] = (15, 24)
    else:
        insights['optimal_ranges'] = {
            'ord2': (10, 19),
            'ord3': (15, 24),
            'ord5': (30, 39),
            'ord6': (40, 45),
        }

    return insights


def generate_firstend_pairs_silent(data):
    """(ord1, ord6) 쌍 생성 (출력 없이)"""
    pair_rounds = defaultdict(list)

    for r in data:
        balls = r['balls']
        ord1, ord6 = balls[0], balls[5]
        pair_rounds[(ord1, ord6)].append(r['round'])

    all_pairs = []
    for ord1 in range(1, 22):
        for ord6 in range(max(ord1 + 5, 23), 46):
            rounds = pair_rounds.get((ord1, ord6), [])
            freq = len(rounds)
            all_pairs.append({
                'ord1': ord1,
                'ord6': ord6,
                'freq': freq,
                'rounds': rounds,
            })

    all_pairs.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord6']))
    return all_pairs


def fill_ord4_silent(pairs):
    """ord4 채우기 (출력 없이)"""
    rows = []
    seen = set()

    for pair in pairs:
        ord1 = pair['ord1']
        ord6 = pair['ord6']
        freq = pair['freq']
        rounds = pair['rounds']

        base_ord4 = round(ord1 + (ord6 - ord1) * FORMULA_RATIO)

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


def fill_ord235_silent(rows, pos_freq, all_freq, recent_3, insights, top_n=5):
    """ord2, ord3, ord5 채우기 (출력 없이)"""
    new_rows = []

    for row in rows:
        ord1 = row['ord1']
        ord4 = row['ord4']
        ord6 = row['ord6']

        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        ord2_picks = find_top_candidates(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3, insights, top_n)
        if not ord2_picks:
            continue

        ord5_candidates = list(range(ord4 + 1, ord6))
        ord5_picks = find_top_candidates(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3, insights, top_n)
        if not ord5_picks:
            continue

        for ord2 in ord2_picks:
            ord3_candidates = list(range(ord2 + 1, ord4))
            ord3_picks = find_top_candidates(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3, insights, top_n)
            if not ord3_picks:
                continue

            for ord3 in ord3_picks:
                for ord5 in ord5_picks:
                    if not (ord1 < ord2 < ord3 < ord4 < ord5 < ord6):
                        continue

                    combo = {
                        'ord1': ord1,
                        'ord2': ord2,
                        'ord3': ord3,
                        'ord4': ord4,
                        'ord5': ord5,
                        'ord6': ord6,
                        'freq': row['freq'],
                        'rounds': row['rounds'],
                        'offset': row['offset'],
                    }

                    # why.py와 동일: 소수 필터 제거
                    new_rows.append(combo)

    new_rows.sort(key=lambda x: (-x['freq'], x['ord1'], x['ord4'], x['ord6']))
    return new_rows


def save_backtest_results(results):
    """백테스트 결과를 result/backtest.csv에 저장"""
    RESULT_DIR.mkdir(exist_ok=True)
    result_path = RESULT_DIR / "backtest.csv"

    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'best_match', 'has_6_match', 'total_combinations', 'best_prediction'])

        for r in results:
            winning_str = str(r['winning'])
            best_pred = r['best_prediction']
            if best_pred:
                best_pred_str = f"[{best_pred['ord1']},{best_pred['ord2']},{best_pred['ord3']},{best_pred['ord4']},{best_pred['ord5']},{best_pred['ord6']}]"
            else:
                best_pred_str = "[]"

            writer.writerow([
                r['round'],
                winning_str,
                r['best_match'],
                r['has_6_match'],
                r['total_combinations'],
                best_pred_str,
            ])

    print(f"\n[결과 저장] {result_path}")
    return result_path


def run_distribution(all_data, start_round, end_round):
    """매치 분포 분석 - 각 회차별 전체 매치 분포 저장"""
    print("=" * 60)
    print(f"매치 분포 분석: {start_round}회차 ~ {end_round}회차")
    print("=" * 60)

    # 인사이트 로드 (한 번만)
    insights = load_insights_silent()

    results_list = []

    for target_round in range(start_round, end_round + 1):
        # 학습 데이터: 직전 회차까지 누적
        train_data = get_data_until(all_data, target_round)
        winning = get_winning_numbers(all_data, target_round)

        if winning is None:
            continue

        if len(train_data) == 0:
            continue

        # 예측 생성
        pairs = generate_firstend_pairs_silent(train_data)
        rows_146 = fill_ord4_silent(pairs)
        pos_freq, all_freq, recent_3 = get_position_stats(train_data)
        predictions = fill_ord235_silent(rows_146, pos_freq, all_freq, recent_3, insights, top_n=15)

        if not predictions:
            continue

        # 매치 분포 계산
        winning_set = set(winning)
        match_counts = Counter()
        best_match = 0
        best_prediction = None
        best_rank = 0

        for rank, pred in enumerate(predictions, 1):
            pred_set = {pred['ord1'], pred['ord2'], pred['ord3'],
                       pred['ord4'], pred['ord5'], pred['ord6']}
            match_count = len(pred_set & winning_set)
            match_counts[match_count] += 1

            if match_count > best_match:
                best_match = match_count
                best_prediction = pred
                best_rank = rank

        result = {
            'round': target_round,
            'winning': winning,
            'best_match': best_match,
            'best_rank': best_rank,
            'best_prediction': best_prediction,
            'total_combinations': len(predictions),
            'match_counts': dict(match_counts),
        }
        results_list.append(result)

        # 진행상황 출력
        has_6 = "✓" if match_counts.get(6, 0) > 0 else " "
        print(f"[{has_6}] {target_round}회차: 최고 {best_match}개, 조합수={len(predictions):,}")

    # 결과 저장
    save_match_distribution(results_list)

    # 요약 출력
    print("\n" + "=" * 60)
    print("[매치 분포 요약]")
    print("=" * 60)
    print(f"총 회차: {len(results_list)}개")

    # 전체 매치 분포 집계
    total_match_counts = Counter()
    total_combinations = 0
    for r in results_list:
        for match, count in r['match_counts'].items():
            total_match_counts[match] += count
        total_combinations += r['total_combinations']

    print(f"총 조합수: {total_combinations:,}개")
    print(f"\n[전체 매치 분포]")
    for i in range(7):
        count = total_match_counts.get(i, 0)
        pct = count / total_combinations * 100 if total_combinations > 0 else 0
        bar = '█' * int(pct / 5) if pct > 0 else ''
        print(f"  {i}개: {count:,}개 ({pct:.1f}%) {bar}")

    return results_list


def run_backtest(all_data, start_round, end_round):
    """백테스트 실행 - 각 회차별 6개 일치 여부 확인"""
    print("=" * 60)
    print(f"백테스트 모드: {start_round}회차 ~ {end_round}회차")
    print("=" * 60)

    # 인사이트 로드 (한 번만)
    insights = load_insights_silent()

    results = []
    six_match_rounds = []

    for target_round in range(start_round, end_round + 1):
        # 학습 데이터: 직전 회차까지 누적
        train_data = get_data_until(all_data, target_round)
        winning = get_winning_numbers(all_data, target_round)

        if winning is None:
            continue

        if len(train_data) == 0:
            continue

        # 예측 생성
        pairs = generate_firstend_pairs_silent(train_data)
        rows_146 = fill_ord4_silent(pairs)
        pos_freq, all_freq, recent_3 = get_position_stats(train_data)
        predictions = fill_ord235_silent(rows_146, pos_freq, all_freq, recent_3, insights, top_n=15)

        if not predictions:
            continue

        # 6개 일치 확인
        winning_set = set(winning)
        has_6_match = False
        best_match = 0
        best_pred = None

        for pred in predictions:
            pred_set = {pred['ord1'], pred['ord2'], pred['ord3'],
                       pred['ord4'], pred['ord5'], pred['ord6']}
            match_count = len(pred_set & winning_set)
            if match_count == 6:
                has_6_match = True
            if match_count > best_match:
                best_match = match_count
                best_pred = pred

        results.append({
            'round': target_round,
            'winning': winning,
            'best_match': best_match,
            'has_6_match': 'Y' if has_6_match else 'N',
            'total_combinations': len(predictions),
            'best_prediction': best_pred,
        })

        # 진행상황 출력
        status = "✓" if has_6_match else " "
        print(f"[{status}] {target_round}회차: {best_match}개 적중, 조합수={len(predictions)}")

        if has_6_match:
            six_match_rounds.append(target_round)

    # 요약 출력
    print("\n" + "=" * 60)
    print("[백테스트 결과 요약]")
    print("=" * 60)
    print(f"총 회차: {len(results)}개")
    print(f"6개 일치: {len(six_match_rounds)}회 ({len(six_match_rounds)/len(results)*100:.1f}%)")
    if six_match_rounds:
        print(f"6개 일치 회차: {six_match_rounds}")

    # 매치 분포
    match_dist = Counter(r['best_match'] for r in results)
    print(f"\n[매치 분포]")
    for i in range(7):
        count = match_dist.get(i, 0)
        pct = count / len(results) * 100 if results else 0
        bar = '█' * int(pct / 2)
        print(f"  {i}개: {count:3d}회 ({pct:5.1f}%) {bar}")

    # 결과 저장
    save_backtest_results(results)

    return results


# ============================================================
# 5개 일치 확장 모드
# ============================================================

def extract_5match_combinations(predictions, winning):
    """5개 일치 조합 추출 및 놓친 포지션 식별

    Returns:
        list[dict]: 각 5개 일치 조합 정보
    """
    winning_set = set(winning)
    position_names = ['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6']

    five_match_combos = []

    for pred in predictions:
        pred_list = [pred['ord1'], pred['ord2'], pred['ord3'],
                     pred['ord4'], pred['ord5'], pred['ord6']]
        pred_set = set(pred_list)

        matched = pred_set & winning_set
        if len(matched) != 5:
            continue

        # 놓친 포지션 찾기 (예측값이 당첨번호에 없는 위치)
        for i, (pos_name, pred_val) in enumerate(zip(position_names, pred_list)):
            if pred_val not in winning_set:
                five_match_combos.append({
                    'combo': pred_list,
                    'matched_values': sorted(matched),
                    'missed_position': pos_name,
                    'missed_idx': i,
                    'predicted_value': pred_val,
                    'actual_value': winning[i],
                })
                break

    return five_match_combos


def expand_5match_to_6match(five_matches, winning):
    """5개 일치 조합을 확장하여 6개 일치 후보 생성

    각 5개 일치 조합에서 틀린 포지션의 유효 범위 내 모든 후보로 교체
    """
    expanded = []
    winning_set = set(winning)

    for item in five_matches:
        combo = item['combo']
        missed_pos = item['missed_position']
        missed_idx = item['missed_idx']
        original_val = item['predicted_value']

        # 포지션별 유효 범위 계산
        ord1, ord2, ord3, ord4, ord5, ord6 = combo

        if missed_pos == 'ord1':
            candidates = range(1, ord2)  # 1 ~ ord2-1
        elif missed_pos == 'ord2':
            candidates = range(ord1 + 1, ord3)  # ord1+1 ~ ord3-1
        elif missed_pos == 'ord3':
            candidates = range(ord2 + 1, ord4)  # ord2+1 ~ ord4-1
        elif missed_pos == 'ord4':
            candidates = range(ord3 + 1, ord5)  # ord3+1 ~ ord5-1
        elif missed_pos == 'ord5':
            candidates = range(ord4 + 1, ord6)  # ord4+1 ~ ord6-1
        elif missed_pos == 'ord6':
            candidates = range(ord5 + 1, 46)  # ord5+1 ~ 45
        else:
            continue

        # 각 후보로 교체한 조합 생성
        for new_val in candidates:
            if new_val == original_val:
                continue  # 원래 값은 스킵

            # 새 조합 생성
            new_combo = combo.copy()
            new_combo[missed_idx] = new_val

            # 6개 일치 여부 확인
            is_6match = set(new_combo) == winning_set

            expanded.append({
                'original_combo': combo,
                'expanded_combo': new_combo,
                'expanded_position': missed_pos,
                'original_value': original_val,
                'new_value': new_val,
                'is_6match': is_6match,
            })

    return expanded


def save_5match_csv(all_5matches, result_path):
    """5개 일치 조합을 CSV로 저장"""
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'combo', 'missed_position',
                         'predicted_value', 'actual_value'])

        for item in all_5matches:
            writer.writerow([
                item['round'],
                str(item['winning']),
                str(item['combo']),
                item['missed_position'],
                item['predicted_value'],
                item['actual_value'],
            ])

    return result_path


def save_expanded_csv(all_expanded, result_path):
    """확장 조합을 CSV로 저장"""
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'original_combo', 'expanded_combo',
                         'expanded_position', 'original_value', 'new_value', 'is_6match'])

        for item in all_expanded:
            writer.writerow([
                item['round'],
                str(item['winning']),
                str(item['original_combo']),
                str(item['expanded_combo']),
                item['expanded_position'],
                item['original_value'],
                item['new_value'],
                'Y' if item['is_6match'] else 'N',
            ])

    return result_path


def save_expansion_backtest_csv(results, result_path):
    """확장 백테스트 결과를 CSV로 저장"""
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', '5match_count', 'expanded_count',
                         'original_has_6match', 'expanded_has_6match', 'improved'])

        for r in results:
            writer.writerow([
                r['round'],
                str(r['winning']),
                r['5match_count'],
                r['expanded_count'],
                'Y' if r['original_has_6match'] else 'N',
                'Y' if r['expanded_has_6match'] else 'N',
                'Y' if r['improved'] else 'N',
            ])

    return result_path


def save_expansion_full_csv(all_expanded, result_path):
    """모든 확장 조합을 CSV로 저장 (단일 회차용)"""
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'winning', 'original_combo', 'expanded_combo',
                         'expanded_position', 'original_value', 'new_value', 'is_6match'])

        for item in all_expanded:
            writer.writerow([
                item['round'],
                str(item['winning']),
                str(item['original_combo']),
                str(item['expanded_combo']),
                item['expanded_position'],
                item['original_value'],
                item['new_value'],
                'Y' if item['is_6match'] else 'N',
            ])

    return result_path


def run_expansion_single(all_data, target_round):
    """단일 회차 확장 백테스트 - 모든 확장 조합을 CSV로 저장"""
    print("=" * 60)
    print(f"확장 백테스트: {target_round}회차")
    print("=" * 60)

    # 인사이트 로드
    insights = load_insights_silent()

    # 학습 데이터
    train_data = get_data_until(all_data, target_round)
    winning = get_winning_numbers(all_data, target_round)

    if winning is None:
        print(f"오류: {target_round}회차 당첨번호가 없습니다.")
        return None

    if len(train_data) == 0:
        print(f"오류: 학습 데이터가 없습니다.")
        return None

    print(f"\n[데이터]")
    print(f"  당첨번호: {winning}")
    print(f"  학습 데이터: {len(train_data)}회차")

    # 예측 생성
    print(f"\n[예측 생성]")
    pairs = generate_firstend_pairs_silent(train_data)
    rows_146 = fill_ord4_silent(pairs)
    pos_freq, all_freq, recent_3 = get_position_stats(train_data)
    predictions = fill_ord235_silent(rows_146, pos_freq, all_freq, recent_3, insights, top_n=15)

    if not predictions:
        print("오류: 예측 조합이 생성되지 않았습니다.")
        return None

    print(f"  총 예측 조합: {len(predictions):,}개")

    # 원본 6개 일치 확인
    winning_set = set(winning)
    original_6match_count = 0
    for pred in predictions:
        pred_set = {pred['ord1'], pred['ord2'], pred['ord3'],
                   pred['ord4'], pred['ord5'], pred['ord6']}
        if len(pred_set & winning_set) == 6:
            original_6match_count += 1

    print(f"  원본 6개 일치: {original_6match_count}개")

    # 5개 일치 조합 추출
    five_matches = extract_5match_combinations(predictions, winning)
    print(f"  5개 일치 조합: {len(five_matches)}개")

    # 확장 조합 생성
    expanded = expand_5match_to_6match(five_matches, winning)

    # 회차/당첨번호 정보 추가
    for item in five_matches:
        item['round'] = target_round
        item['winning'] = winning

    for item in expanded:
        item['round'] = target_round
        item['winning'] = winning

    # 6개 일치 확장 조합 수
    expanded_6match_count = sum(1 for e in expanded if e['is_6match'])

    print(f"\n[확장 결과]")
    print(f"  총 확장 조합: {len(expanded)}개")
    print(f"  확장 중 6개 일치: {expanded_6match_count}개")

    # CSV 저장
    RESULT_DIR.mkdir(exist_ok=True)

    save_5match_csv(five_matches, RESULT_DIR / "5match_combos.csv")
    save_expansion_full_csv(expanded, RESULT_DIR / "expansion_backtest.csv")

    print(f"\n[저장된 파일]")
    print(f"  - result/5match_combos.csv ({len(five_matches)}개)")
    print(f"  - result/expansion_backtest.csv ({len(expanded)}개)")

    return expanded


def run_expansion_backtest(all_data, start_round, end_round):
    """확장 백테스트 실행 - 5개 일치 조합 확장하여 6개 일치 복구 테스트"""
    print("=" * 60)
    print(f"확장 백테스트: {start_round}회차 ~ {end_round}회차")
    print("=" * 60)

    # 인사이트 로드
    insights = load_insights_silent()

    results = []
    all_5matches = []
    all_expanded = []

    original_6match_count = 0
    expanded_6match_count = 0
    improved_count = 0

    for target_round in range(start_round, end_round + 1):
        # 학습 데이터
        train_data = get_data_until(all_data, target_round)
        winning = get_winning_numbers(all_data, target_round)

        if winning is None or len(train_data) == 0:
            continue

        # 예측 생성
        pairs = generate_firstend_pairs_silent(train_data)
        rows_146 = fill_ord4_silent(pairs)
        pos_freq, all_freq, recent_3 = get_position_stats(train_data)
        predictions = fill_ord235_silent(rows_146, pos_freq, all_freq, recent_3, insights, top_n=15)

        if not predictions:
            continue

        # 원본 6개 일치 확인
        winning_set = set(winning)
        original_has_6match = False
        for pred in predictions:
            pred_set = {pred['ord1'], pred['ord2'], pred['ord3'],
                       pred['ord4'], pred['ord5'], pred['ord6']}
            if len(pred_set & winning_set) == 6:
                original_has_6match = True
                break

        # 5개 일치 조합 추출
        five_matches = extract_5match_combinations(predictions, winning)

        # 확장 조합 생성
        expanded = expand_5match_to_6match(five_matches, winning)

        # 확장 후 6개 일치 확인
        expanded_has_6match = original_has_6match or any(e['is_6match'] for e in expanded)

        # 개선 여부
        improved = expanded_has_6match and not original_has_6match

        # 통계 업데이트
        if original_has_6match:
            original_6match_count += 1
        if expanded_has_6match:
            expanded_6match_count += 1
        if improved:
            improved_count += 1

        # 결과 저장
        results.append({
            'round': target_round,
            'winning': winning,
            '5match_count': len(five_matches),
            'expanded_count': len(expanded),
            'original_has_6match': original_has_6match,
            'expanded_has_6match': expanded_has_6match,
            'improved': improved,
        })

        # 5개 일치 조합에 회차 정보 추가
        for item in five_matches:
            item['round'] = target_round
            item['winning'] = winning
            all_5matches.append(item)

        # 확장 조합에 회차 정보 추가
        for item in expanded:
            item['round'] = target_round
            item['winning'] = winning
            all_expanded.append(item)

        # 진행상황 출력
        orig_mark = "✓" if original_has_6match else " "
        exp_mark = "✓" if expanded_has_6match else " "
        imp_mark = "↑" if improved else " "
        print(f"[{orig_mark}→{exp_mark}]{imp_mark} {target_round}회차: "
              f"5매치={len(five_matches)}, 확장={len(expanded)}")

    # CSV 저장
    RESULT_DIR.mkdir(exist_ok=True)

    save_5match_csv(all_5matches, RESULT_DIR / "5match_combos.csv")
    save_expanded_csv(all_expanded, RESULT_DIR / "expanded_combos.csv")
    save_expansion_backtest_csv(results, RESULT_DIR / "expansion_backtest.csv")

    # 요약 출력
    total_rounds = len(results)
    print("\n" + "=" * 60)
    print("[확장 백테스트 결과]")
    print("=" * 60)
    print(f"총 회차: {total_rounds}개")
    print(f"원본 6개 일치: {original_6match_count}회 ({original_6match_count/total_rounds*100:.1f}%)")
    print(f"확장 6개 일치: {expanded_6match_count}회 ({expanded_6match_count/total_rounds*100:.1f}%)")
    print(f"개선된 회차: {improved_count}회")
    print(f"\n총 5개 일치 조합: {len(all_5matches)}개")
    print(f"총 확장 조합: {len(all_expanded)}개")
    print(f"확장 중 6개 일치: {sum(1 for e in all_expanded if e['is_6match'])}개")

    print(f"\n[저장된 파일]")
    print(f"  - result/5match_combos.csv")
    print(f"  - result/expanded_combos.csv")
    print(f"  - result/expansion_backtest.csv")

    return results


# ============================================================
# 클러스터링 모드
# ============================================================

def run_cluster_mode(all_data, args):
    """5개 공유 그룹 클러스터링 실행"""
    from importlib import import_module

    # cluster 모듈 동적 로드
    sys.path.insert(0, str(BASE_DIR / "5_cluster"))
    import cluster as cluster_module

    # 범위 백테스트
    if args.start != 900 or args.end != 1000:
        print("=" * 60)
        print(f"클러스터 백테스트: {args.start}회차 ~ {args.end}회차")
        print("=" * 60)

        results = []

        for target_round in range(args.start, args.end + 1):
            winning = get_winning_numbers(all_data, target_round)
            if winning is None:
                continue

            # 예측 생성
            predictions = generate_predictions_silent(all_data, target_round)
            if not predictions:
                continue

            # 클러스터링
            cluster_counts = cluster_module.cluster_combinations(predictions)
            top_clusters = cluster_module.get_top_clusters(cluster_counts, 100)

            # 당첨번호 체크
            match_5 = 0
            best_rank = None

            for rank, (five_key, count) in enumerate(top_clusters, 1):
                check = cluster_module.check_winning_in_cluster(five_key, count, winning)
                if check['has_5match']:
                    match_5 += 1
                    if best_rank is None:
                        best_rank = rank

            results.append({
                'round': target_round,
                'match_5': match_5,
                'best_rank': best_rank,
            })

            status = "✓" if match_5 > 0 else " "
            rank_str = f"rank={best_rank}" if best_rank else "N/A"
            print(f"[{status}] {target_round}회차: 5매치={match_5}, {rank_str}")

        # 요약
        total = len(results)
        has_5 = sum(1 for r in results if r['match_5'] > 0)

        print("\n" + "=" * 60)
        print(f"[백테스트 결과]")
        print(f"  총 회차: {total}개")
        print(f"  Top-100 중 5개 일치 포함: {has_5}회 ({has_5/total*100:.1f}%)")

    else:
        # 단일 회차 클러스터링
        target_round = args.round

        print("=" * 60)
        print(f"클러스터링: {target_round}회차")
        print("=" * 60)

        winning = get_winning_numbers(all_data, target_round)

        # 예측 생성
        predictions = generate_predictions_silent(all_data, target_round)
        if not predictions:
            print("오류: 예측 조합이 생성되지 않았습니다.")
            return

        # 클러스터링 실행
        top_clusters = cluster_module.run_clustering(predictions, winning, top_n=100)

        print(f"\n[저장] 5_cluster/result/clusters.csv")


def generate_predictions_silent(all_data, target_round):
    """예측 조합 생성 (출력 없음)"""
    train_data = get_data_until(all_data, target_round)
    if len(train_data) == 0:
        return None

    insights = load_insights_silent()
    pairs = generate_firstend_pairs_silent(train_data)
    rows_146 = fill_ord4_silent(pairs)
    pos_freq, all_freq, recent_3 = get_position_stats(train_data)
    predictions = fill_ord235_silent(rows_146, pos_freq, all_freq, recent_3, insights, top_n=15)

    return predictions


# ============================================================
# 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='로또 예측 시스템')
    parser.add_argument('--round', type=int, default=1204, help='목표 회차 (기본: 1204)')
    parser.add_argument('--backtest', action='store_true', help='백테스트 모드')
    parser.add_argument('--distribution', action='store_true', help='매치 분포 분석 모드')
    parser.add_argument('--expansion', action='store_true', help='5개 일치 → 6개 확장 백테스트')
    parser.add_argument('--cluster', action='store_true', help='5개 공유 그룹 클러스터링')
    parser.add_argument('--start', type=int, default=900, help='시작 회차')
    parser.add_argument('--end', type=int, default=1000, help='종료 회차')
    args = parser.parse_args()

    # 데이터 로드
    all_data = load_all_data()

    # 클러스터링 모드
    if args.cluster:
        run_cluster_mode(all_data, args)
        return

    # 5개 일치 확장 백테스트 모드
    if args.expansion:
        # --start/--end 지정 시 범위 백테스트, 아니면 --round로 단일 회차
        if args.start != 900 or args.end != 1000:
            run_expansion_backtest(all_data, args.start, args.end)
        else:
            run_expansion_single(all_data, args.round)
        return

    # 매치 분포 분석 모드
    if args.distribution:
        run_distribution(all_data, args.start, args.end)
        return

    # 백테스트 모드
    if args.backtest:
        run_backtest(all_data, args.start, args.end)
        return

    # 단일 회차 예측 모드
    target_round = args.round

    print("=" * 60)
    print(f"로또 예측 시스템 - 목표 회차: {target_round}")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[데이터 로드]")
    print(f"  전체 데이터: {len(all_data)}회차")
    print(f"  데이터 범위: {all_data[0]['round']} ~ {all_data[-1]['round']}")

    # 목표 회차 직전까지 데이터
    train_data = get_data_until(all_data, target_round)
    print(f"  학습 데이터: {len(train_data)}회차 (목표 직전까지)")

    if len(train_data) == 0:
        print(f"\n오류: 목표 회차 {target_round}에 대한 학습 데이터가 없습니다.")
        sys.exit(1)

    # 당첨번호 확인 (있으면)
    winning = get_winning_numbers(all_data, target_round)

    # 2. 인사이트 학습
    run_insights(target_round, all_data)

    # 3. 인사이트 로드 (동적)
    print("\n[인사이트 로드]")
    insights = load_insights()

    # 4. FirstEnd: (ord1, ord6) 쌍 생성
    pairs = generate_firstend_pairs(train_data)

    # 5. ord4 계산
    rows_146 = fill_ord4(pairs)

    # 6. ord235 채우기
    pos_freq, all_freq, recent_3 = get_position_stats(train_data)
    predictions = fill_ord235(rows_146, pos_freq, all_freq, recent_3, insights, top_n=15)

    if not predictions:
        print("\n오류: 유효한 예측 조합이 생성되지 않았습니다.")
        sys.exit(1)

    # 7. 당첨번호 검증 (결과 저장 전)
    check_winning_match(predictions, winning, target_round)

    # 8. 결과 저장
    save_result(predictions, target_round)

    # 요약 출력
    print_summary(predictions, target_round)


if __name__ == "__main__":
    main()
