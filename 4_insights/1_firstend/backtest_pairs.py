"""
첫수/끝수 점수 기반 선택 백테스트

각 회차마다:
1. 해당 회차 이전 데이터로 통계 생성
2. 점수 기반으로 상위 90% 선택
3. 실제 당첨번호의 (첫수,끝수)가 선택에 포함되는지 확인
"""

import csv
from pathlib import Path
from collections import Counter

DATA_PATH = Path(__file__).parent.parent.parent / "1_data" / "winning_numbers.csv"


def load_all_data() -> list:
    """전체 당첨번호 데이터 로드"""
    results = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            balls = sorted([int(row[f'ball{i}']) for i in range(1, 7)])
            results.append({
                'round': int(row['round']),
                'balls': balls,
                'first': balls[0],
                'end': balls[-1]
            })
    return results


def build_distributions(data: list) -> tuple:
    """데이터로부터 분포 생성"""
    first_dist = Counter(r['first'] for r in data)
    end_dist = Counter(r['end'] for r in data)
    pair_dist = Counter((r['first'], r['end']) for r in data)
    return first_dist, end_dist, pair_dist


def calculate_scores(first_dist: dict, end_dist: dict, pair_dist: dict, recent_firstend: set, penalty: int = 5) -> list:
    """점수 계산"""
    scores = []

    for first in range(1, 31):
        for end in range(max(first + 7, 17), 46):
            first_score = first_dist.get(first, 0)
            end_score = end_dist.get(end, 0)
            pair_score = pair_dist.get((first, end), 0)

            total = first_score + end_score + pair_score

            if first in recent_firstend:
                total -= penalty
            if end in recent_firstend:
                total -= penalty

            scores.append({
                'first': first,
                'end': end,
                'total': total
            })

    scores.sort(key=lambda x: x['total'], reverse=True)
    return scores


def select_top_90_percent(scores: list) -> set:
    """상위 90% 조합 선택"""
    total_sum = sum(s['total'] for s in scores if s['total'] > 0)

    selected = set()
    cumsum = 0
    for s in scores:
        if s['total'] <= 0:
            continue
        cumsum += s['total']
        selected.add((s['first'], s['end']))
        if cumsum >= total_sum * 0.9:
            break

    return selected


def backtest(start_round: int = 876):
    """백테스트 실행"""
    all_data = load_all_data()

    # 시작 회차 인덱스 찾기
    start_idx = None
    for i, d in enumerate(all_data):
        if d['round'] == start_round:
            start_idx = i
            break

    if start_idx is None:
        print(f"회차 {start_round}를 찾을 수 없습니다.")
        return

    results = []
    hits = 0
    total = 0

    for i in range(start_idx, len(all_data)):
        target = all_data[i]
        target_round = target['round']
        target_first = target['first']
        target_end = target['end']

        # 이전 데이터만 사용
        past_data = all_data[:i]

        if len(past_data) < 50:  # 최소 50회차 이상 필요
            continue

        # 분포 생성
        first_dist, end_dist, pair_dist = build_distributions(past_data)

        # 최근 3회 첫수/끝수
        recent = set()
        for r in past_data[-3:]:
            recent.add(r['first'])
            recent.add(r['end'])

        # 점수 계산 및 선택
        scores = calculate_scores(first_dist, end_dist, pair_dist, recent)
        selected = select_top_90_percent(scores)

        # 적중 확인
        hit = (target_first, target_end) in selected
        if hit:
            hits += 1
        total += 1

        results.append({
            'round': target_round,
            'first': target_first,
            'end': target_end,
            'selected_count': len(selected),
            'hit': hit
        })

    # 결과 출력
    print("=== 첫수/끝수 백테스트 결과 ===\n")
    print(f"테스트 범위: {start_round}회차 ~ {all_data[-1]['round']}회차")
    print(f"총 테스트: {total}회차")
    print(f"적중: {hits}회")
    print(f"적중률: {hits/total*100:.1f}%\n")

    # 최근 20회 상세
    print("최근 20회차 결과:")
    print("-" * 60)
    print(f"{'회차':>6} {'첫수':>4} {'끝수':>4} {'선택수':>6} {'적중':>6}")
    print("-" * 60)
    for r in results[-20:]:
        hit_mark = "O" if r['hit'] else "X"
        print(f"{r['round']:>6} {r['first']:>4} {r['end']:>4} {r['selected_count']:>6} {hit_mark:>6}")

    # 미적중 분석
    misses = [r for r in results if not r['hit']]
    print(f"\n미적중 회차 수: {len(misses)}회")

    if misses:
        print("\n미적중 회차 (최근 10개):")
        for r in misses[-10:]:
            print(f"  {r['round']}회차: 첫수={r['first']}, 끝수={r['end']}")

    # 결과 저장
    output_path = Path(__file__).parent / "statistics" / "backtest_result.csv"
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'first', 'end', 'selected_count', 'hit'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    backtest()
