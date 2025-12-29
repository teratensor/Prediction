"""
(ord1, ord6) 쌍 빈도 결과 내보내기

실행: python export_result.py
출력: result/result.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR.parent.parent / "1_data" / "winning_numbers.csv"
RESULT_DIR = BASE_DIR.parent.parent / "result"


def load_data():
    """당첨번호 로드 - (ord1, ord6) 쌍과 회차 추출"""
    pair_rounds = defaultdict(list)  # {(ord1, ord6): [회차1, 회차2, ...]}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            round_num = int(row['round'])
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            ord1 = balls[0]   # 첫수
            ord6 = balls[5]   # 끝수
            pair_rounds[(ord1, ord6)].append(round_num)
    return pair_rounds


def export_pair_frequency(pair_rounds):
    """(ord1, ord6) 쌍 빈도 저장 - 모든 가능한 쌍 포함"""
    RESULT_DIR.mkdir(exist_ok=True)

    total = sum(len(rounds) for rounds in pair_rounds.values())

    # 모든 가능한 쌍 생성 (ord1 <= 21, ord6 >= 23, 최소 5칸 차이)
    all_pairs = []
    for ord1 in range(1, 22):  # 1~21
        for ord6 in range(max(ord1 + 5, 23), 46):  # ord1+5 ~ 45, ord6 >= 23
            rounds = pair_rounds.get((ord1, ord6), [])
            freq = len(rounds)
            rounds_str = ','.join(map(str, sorted(rounds))) if rounds else ''
            all_pairs.append((ord1, ord6, freq, rounds_str))

    # 빈도순 정렬 (높은 순), 같으면 ord1 오름차순, ord6 오름차순
    all_pairs.sort(key=lambda x: (-x[2], x[0], x[1]))

    # CSV 저장
    with open(RESULT_DIR / "result.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6', '빈도수', '회차'])

        for ord1, ord6, freq, rounds_str in all_pairs:
            writer.writerow([ord1, '', '', '', '', ord6, freq, rounds_str])

    seen_pairs = sum(1 for _, _, f, _ in all_pairs if f > 0)
    unseen_pairs = sum(1 for _, _, f, _ in all_pairs if f == 0)

    print(f"총 {total}회차 분석")
    print(f"전체 가능한 쌍: {len(all_pairs)}개")
    print(f"출현 쌍: {seen_pairs}개")
    print(f"미출현 쌍: {unseen_pairs}개")
    print(f"저장 완료: {RESULT_DIR / 'result.csv'}")

    # 상위 10개 출력
    print("\n[상위 10개 쌍]")
    sorted_pairs = sorted(pair_rounds.items(), key=lambda x: -len(x[1]))
    for i, ((ord1, ord6), rounds) in enumerate(sorted_pairs[:10], 1):
        print(f"  {i}. ({ord1}, {ord6}): {len(rounds)}회 - {rounds}")


if __name__ == "__main__":
    pair_rounds = load_data()
    export_pair_frequency(pair_rounds)
