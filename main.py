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

    # 3. 최빈 구간 로드
    range_path = INSIGHTS_DIR / "4_range" / "statistics" / "position_range_distribution.csv"
    if range_path.exists():
        best_range = {}
        with open(range_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos = int(row['position'])
                prob = float(row['probability'])
                if pos in [2, 3, 5]:
                    if pos not in best_range or prob > best_range[pos][1]:
                        range_idx = int(row['range'])
                        best_range[pos] = (range_idx, prob)

        # range_idx → (min, max) 변환
        range_map = {0: (1, 9), 1: (10, 19), 2: (20, 29), 3: (30, 39), 4: (40, 45)}
        for pos, (range_idx, _) in best_range.items():
            insights['optimal_ranges'][f'ord{pos}'] = range_map[range_idx]
        print(f"    - 4_range: {insights['optimal_ranges']}")
    else:
        print(f"    - 4_range: 파일 없음 (기본값 사용)")
        insights['optimal_ranges'] = {'ord2': (10, 19), 'ord3': (10, 19), 'ord5': (30, 39)}

    return insights


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
    """후보 번호 점수 계산 - 인사이트 사용"""
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

    if num in insights['hot_bits']:
        score += 5
    if num in insights['cold_bits']:
        score -= 3
    if num in recent_3:
        score -= 5
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


def fill_ord235(rows, pos_freq, all_freq, recent_3, insights, top_n=15):
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

                    new_rows.append({
                        'ord1': ord1,
                        'ord2': ord2,
                        'ord3': ord3,
                        'ord4': ord4,
                        'ord5': ord5,
                        'ord6': ord6,
                        'freq': row['freq'],
                        'rounds': row['rounds'],
                        'offset': row['offset'],
                    })

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
        'best_match': best_match,
        'best_rank': best_rank,
        'match_counts': dict(match_counts),
    }


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
# 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='로또 예측 시스템')
    parser.add_argument('--round', type=int, default=1204, help='목표 회차 (기본: 1204)')
    args = parser.parse_args()

    target_round = args.round

    print("=" * 60)
    print(f"로또 예측 시스템 - 목표 회차: {target_round}")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[데이터 로드]")
    all_data = load_all_data()
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
    predictions = fill_ord235(rows_146, pos_freq, all_freq, recent_3, insights)

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
