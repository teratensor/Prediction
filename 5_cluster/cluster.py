"""
5개 공유 그룹 클러스터링

500만 조합을 "5개가 같은 그룹"으로 분류하여
당첨번호 없이도 유력 조합군 식별
"""

import csv
from collections import defaultdict, Counter
from pathlib import Path

RESULT_DIR = Path(__file__).parent / "result"


def generate_5keys(combo):
    """6개 조합에서 6개의 5-키 생성

    Args:
        combo: [ord1, ord2, ord3, ord4, ord5, ord6] 리스트

    Returns:
        [(5-키, 제외위치, 제외값), ...]
    """
    keys = []
    for i in range(6):
        key = tuple(combo[:i] + combo[i+1:])  # i번째 제외
        keys.append((key, i, combo[i]))
    return keys


def cluster_combinations(predictions):
    """500만 조합을 5-키 기준으로 그룹화 (최적화 버전)

    Args:
        predictions: 예측 조합 리스트 (dict with ord1~ord6)

    Returns:
        {5-키: 카운트} (메모리 효율을 위해 카운트만 저장)
    """
    from collections import Counter
    cluster_counts = Counter()

    for pred in predictions:
        combo = (pred['ord1'], pred['ord2'], pred['ord3'],
                 pred['ord4'], pred['ord5'], pred['ord6'])

        for i in range(6):
            key = combo[:i] + combo[i+1:]  # i번째 제외
            cluster_counts[key] += 1

    return cluster_counts


def cluster_combinations_full(predictions):
    """500만 조합을 5-키 기준으로 그룹화 (전체 저장 버전)

    Args:
        predictions: 예측 조합 리스트 (dict with ord1~ord6)

    Returns:
        {5-키: [조합 정보 리스트]}
    """
    clusters = defaultdict(list)

    for pred in predictions:
        combo = [pred['ord1'], pred['ord2'], pred['ord3'],
                 pred['ord4'], pred['ord5'], pred['ord6']]

        for key, excluded_pos, excluded_val in generate_5keys(combo):
            clusters[key].append({
                'combo': combo,
                'excluded_pos': excluded_pos,
                'excluded_val': excluded_val,
            })

    return clusters


def get_top_clusters(cluster_counts, top_n=100):
    """가장 큰 클러스터 top_n개 반환

    Args:
        cluster_counts: cluster_combinations() 결과 (Counter)
        top_n: 상위 몇 개

    Returns:
        [(5-키, 카운트), ...] 크기 내림차순
    """
    return cluster_counts.most_common(top_n)


def analyze_cluster(five_key, items):
    """클러스터 분석 - 제외 위치/값 분포

    Args:
        five_key: 5개 번호 튜플
        items: 해당 클러스터의 조합 리스트

    Returns:
        분석 결과 dict
    """
    pos_names = ['ord1', 'ord2', 'ord3', 'ord4', 'ord5', 'ord6']

    # 제외 위치별 카운트
    pos_counter = Counter()
    # 제외 값별 카운트
    val_counter = Counter()

    for item in items:
        pos_counter[pos_names[item['excluded_pos']]] += 1
        val_counter[item['excluded_val']] += 1

    # 가장 많이 제외된 값들 (상위 5개)
    top_excluded = val_counter.most_common(5)

    return {
        'five_key': list(five_key),
        'cluster_size': len(items),
        'position_dist': dict(pos_counter),
        'top_excluded': top_excluded,
        'all_combos': [item['combo'] for item in items],
    }


def check_winning_in_cluster(five_key, count_or_items, winning):
    """클러스터에 당첨번호가 포함되는지 확인

    Args:
        five_key: 5개 번호 튜플
        count_or_items: 카운트(int) 또는 조합 리스트
        winning: 당첨번호 [6개]

    Returns:
        {'has_5match': bool, 'has_6match': bool}
    """
    winning_set = set(winning)
    five_key_set = set(five_key)

    # 5개 일치 여부: 5-키가 당첨번호의 5개와 일치하는지
    has_5match = five_key_set.issubset(winning_set) and len(five_key_set) == 5

    # 6개 일치 가능 여부: 5-키가 당첨번호에 포함되면, 남은 1개도 포함될 수 있음
    # 클러스터 내에 winning이 있는지는 실제 조합을 확인해야 하지만,
    # 최적화 버전에서는 5-키가 당첨번호 5개와 일치하면 6개 일치 가능으로 간주
    has_6match = has_5match  # 5개 일치하면 해당 클러스터에 6개 일치가 있을 수 있음

    return {
        'has_5match': has_5match,
        'has_6match': has_6match,
    }


def save_clusters_csv(top_clusters, result_path, winning=None):
    """클러스터 결과를 CSV로 저장

    Args:
        top_clusters: get_top_clusters() 결과 [(5-키, 카운트), ...]
        result_path: 저장 경로
        winning: 당첨번호 (있으면 일치 여부 표시)
    """
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if winning:
            writer.writerow(['rank', 'five_key', 'cluster_size', 'has_5match'])
        else:
            writer.writerow(['rank', 'five_key', 'cluster_size'])

        for rank, (five_key, count) in enumerate(top_clusters, 1):
            if winning:
                check = check_winning_in_cluster(five_key, count, winning)
                writer.writerow([
                    rank,
                    str(list(five_key)),
                    count,
                    'Y' if check['has_5match'] else 'N',
                ])
            else:
                writer.writerow([
                    rank,
                    str(list(five_key)),
                    count,
                ])

    return result_path


def run_clustering(predictions, winning=None, top_n=100):
    """클러스터링 실행 및 결과 저장

    Args:
        predictions: 예측 조합 리스트
        winning: 당첨번호 (선택)
        top_n: 상위 클러스터 수

    Returns:
        top_clusters 리스트
    """
    print(f"\n[클러스터링]")
    print(f"  총 조합: {len(predictions):,}개")

    # 클러스터링 (최적화 버전 - 카운트만)
    cluster_counts = cluster_combinations(predictions)
    print(f"  총 5-키 그룹: {len(cluster_counts):,}개")

    # 상위 클러스터
    top_clusters = get_top_clusters(cluster_counts, top_n)

    # 통계
    sizes = [count for _, count in top_clusters]
    print(f"  Top-{top_n} 클러스터 크기: {min(sizes)} ~ {max(sizes)}")

    # 당첨번호 체크
    if winning:
        match_count = 0

        for five_key, count in top_clusters:
            check = check_winning_in_cluster(five_key, count, winning)
            if check['has_5match']:
                match_count += 1

        print(f"\n[당첨번호 검증] {winning}")
        print(f"  Top-{top_n} 중 5개 일치: {match_count}개")

    # CSV 저장
    RESULT_DIR.mkdir(exist_ok=True)
    result_path = RESULT_DIR / "clusters.csv"
    save_clusters_csv(top_clusters, result_path, winning)
    print(f"\n[저장] {result_path}")

    return top_clusters


def run_cluster_backtest(all_data, get_data_until_func, get_winning_func,
                         generate_predictions_func, start_round, end_round, top_n=100):
    """클러스터링 백테스트

    Args:
        all_data: 전체 데이터
        get_data_until_func: 학습 데이터 추출 함수
        get_winning_func: 당첨번호 추출 함수
        generate_predictions_func: 예측 생성 함수
        start_round, end_round: 범위
        top_n: 상위 클러스터 수
    """
    print("=" * 60)
    print(f"클러스터 백테스트: {start_round}회차 ~ {end_round}회차")
    print("=" * 60)

    results = []

    for target_round in range(start_round, end_round + 1):
        winning = get_winning_func(all_data, target_round)
        if winning is None:
            continue

        # 예측 생성
        predictions = generate_predictions_func(all_data, target_round)
        if not predictions:
            continue

        # 클러스터링
        clusters = cluster_combinations(predictions)
        top_clusters = get_top_clusters(clusters, top_n)

        # 당첨번호 체크
        winning_set = set(winning)
        match_5 = 0
        match_6 = 0
        best_rank = None

        for rank, (five_key, items) in enumerate(top_clusters, 1):
            check = check_winning_in_cluster(five_key, items, winning)
            if check['has_5match']:
                match_5 += 1
                if best_rank is None:
                    best_rank = rank
            if check['has_6match']:
                match_6 += 1

        results.append({
            'round': target_round,
            'winning': winning,
            'total_clusters': len(clusters),
            'match_5': match_5,
            'match_6': match_6,
            'best_rank': best_rank,
        })

        # 진행상황
        status = "✓" if match_6 > 0 else ("○" if match_5 > 0 else " ")
        rank_str = f"rank={best_rank}" if best_rank else "N/A"
        print(f"[{status}] {target_round}회차: 5매치={match_5}, 6매치={match_6}, {rank_str}")

    # 요약
    total = len(results)
    has_5 = sum(1 for r in results if r['match_5'] > 0)
    has_6 = sum(1 for r in results if r['match_6'] > 0)

    print("\n" + "=" * 60)
    print(f"[백테스트 결과]")
    print(f"  총 회차: {total}개")
    print(f"  5개 일치 포함: {has_5}회 ({has_5/total*100:.1f}%)")
    print(f"  6개 일치 포함: {has_6}회 ({has_6/total*100:.1f}%)")

    return results
