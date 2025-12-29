"""
ord2, ord3, ord5 채우기 (backtest_v4b 방식)

입력: result/result.csv (ord1, ord4, ord6 결정됨)
출력: result/result.csv (ord2, ord3, ord5 추가)

방식: 순차적 50/50 + 점수화
- 포지션별 최빈 구간 보너스
- 핫/콜드 비트 반영
- 소수 보너스
"""

import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent.parent / "1_data" / "winning_numbers.csv"
RESULT_PATH = BASE_DIR.parent.parent / "result" / "result.csv"

# 포지션별 최빈 구간
OPTIMAL_RANGES = {
    'ord2': (10, 19),  # 48.3%
    'ord3': (10, 19),  # 45.4%
    'ord5': (30, 39),  # 54.6%
}

# 핫/콜드 비트 (12_onehot)
HOT_BITS = {29, 17, 27, 3, 25, 1, 19, 39, 4, 31}
COLD_BITS = {41, 43, 8, 14, 34, 26, 22, 44, 20, 7}

# 소수
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}


def load_winning_data():
    """당첨번호 데이터 로드"""
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


def get_position_stats(data):
    """포지션별 빈도 통계"""
    pos_freq = {
        'ord2': Counter(),  # balls[1]
        'ord3': Counter(),  # balls[2]
        'ord5': Counter(),  # balls[4]
    }
    all_freq = Counter()

    for r in data:
        balls = r['balls']
        pos_freq['ord2'][balls[1]] += 1
        pos_freq['ord3'][balls[2]] += 1
        pos_freq['ord5'][balls[4]] += 1
        for b in balls:
            all_freq[b] += 1

    # 최근 3회 번호
    recent_3 = set()
    for r in data[-3:]:
        recent_3.update(r['balls'])

    return pos_freq, all_freq, recent_3


def score_candidate(num, pos_name, pos_freq, all_freq, recent_3):
    """후보 번호 점수 계산"""
    seen_numbers = set(pos_freq[pos_name].keys())
    optimal_min, optimal_max = OPTIMAL_RANGES[pos_name]

    if num in seen_numbers:
        # seen 번호
        score = pos_freq[pos_name][num] * 10
        if optimal_min <= num <= optimal_max:
            score += 15
    else:
        # unseen 번호
        score = all_freq.get(num, 0) * 2
        if optimal_min <= num <= optimal_max:
            score += 20

    # 핫/콜드 비트
    if num in HOT_BITS:
        score += 5
    if num in COLD_BITS:
        score -= 3

    # 최근 3회 페널티
    if num in recent_3:
        score -= 5

    # 소수 보너스
    if num in PRIMES:
        score += 3

    return score


def find_best_candidate(candidates, pos_name, pos_freq, all_freq, recent_3):
    """범위 내에서 최고 점수 후보 선택"""
    if not candidates:
        return None

    best_num = None
    best_score = -999

    for num in candidates:
        score = score_candidate(num, pos_name, pos_freq, all_freq, recent_3)
        if score > best_score:
            best_score = score
            best_num = num

    return best_num


def load_result():
    """result.csv 로드"""
    rows = []
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def fill_ord235(rows, pos_freq, all_freq, recent_3):
    """ord2, ord3, ord5 채우기"""
    new_rows = []

    for row in rows:
        ord1 = int(row['ord1'])
        ord4 = int(row['ord4'])
        ord6 = int(row['ord6'])

        # ord2 후보: ord1+1 ~ ord4-2
        ord2_candidates = list(range(ord1 + 1, ord4 - 1))
        ord2 = find_best_candidate(ord2_candidates, 'ord2', pos_freq, all_freq, recent_3)

        if ord2 is None:
            continue

        # ord3 후보: ord2+1 ~ ord4-1
        ord3_candidates = list(range(ord2 + 1, ord4))
        ord3 = find_best_candidate(ord3_candidates, 'ord3', pos_freq, all_freq, recent_3)

        if ord3 is None:
            continue

        # ord5 후보: ord4+1 ~ ord6-1
        ord5_candidates = list(range(ord4 + 1, ord6))
        ord5 = find_best_candidate(ord5_candidates, 'ord5', pos_freq, all_freq, recent_3)

        if ord5 is None:
            continue

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
            '빈도수': row['빈도수'],
            '회차': row['회차'],
            'offset': row['offset'],
        })

    return new_rows


def save_result(rows):
    """result.csv 저장"""
    with open(RESULT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6', '빈도수', '회차', 'offset'])

        for row in rows:
            writer.writerow([
                row['ord1'],
                row['ord2'],
                row['ord3'],
                row['ord4'],
                row['ord5'],
                row['ord6'],
                row['빈도수'],
                row['회차'],
                row['offset'],
            ])

    print(f"저장 완료: {RESULT_PATH}")


def main():
    # 1. 당첨번호 데이터 로드
    data = load_winning_data()
    print(f"당첨번호: {len(data)}회차")

    # 2. 통계 계산
    pos_freq, all_freq, recent_3 = get_position_stats(data)

    # 3. result.csv 로드
    rows = load_result()
    print(f"입력: {len(rows)}개 행")

    # 4. ord2, ord3, ord5 채우기
    new_rows = fill_ord235(rows, pos_freq, all_freq, recent_3)
    print(f"출력: {len(new_rows)}개 행")

    # 5. 빈도순 정렬
    new_rows.sort(key=lambda x: (-int(x['빈도수']), x['ord1'], x['ord4'], x['ord6']))

    # 6. 저장
    save_result(new_rows)

    # 7. 샘플 출력
    print("\n[상위 10개 조합]")
    for i, row in enumerate(new_rows[:10], 1):
        print(f"  {i}. ({row['ord1']}, {row['ord2']}, {row['ord3']}, {row['ord4']}, {row['ord5']}, {row['ord6']}) - 빈도 {row['빈도수']}, offset {row['offset']}")


if __name__ == "__main__":
    main()
