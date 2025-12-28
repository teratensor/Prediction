"""
예측 알고리즘 모듈
가중치 기반 점수 시스템으로 로또 번호 예측
Top24, Mid14, Rest7 세그먼트 사용
점수순 정렬 유지 (오름차순 정렬 제거)
"""
import csv
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

# 소수 목록 (1-45 범위)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]


def load_data(data_path: str = None) -> tuple:
    """당첨번호 데이터 로드 (CSV 형식)

    Returns:
        (DataFrame, dict): (당첨번호 DataFrame, {회차: {ord: ball}} o값 매핑)
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"

    rows = []
    o_values_map = {}  # {회차: {ord: ball}}

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            round_num = int(row['round'])
            rows.append({
                '회차': round_num,
                'ord1': int(row['ord1']),
                'ord2': int(row['ord2']),
                'ord3': int(row['ord3']),
                'ord4': int(row['ord4']),
                'ord5': int(row['ord5']),
                'ord6': int(row['ord6']),
                'ball1': int(row['ball1']),
                'ball2': int(row['ball2']),
                'ball3': int(row['ball3']),
                'ball4': int(row['ball4']),
                'ball5': int(row['ball5']),
                'ball6': int(row['ball6'])
            })

            # o값 매핑 생성: o1~o45 컬럼에서 {ord: ball} 추출
            o_values = {}
            for i in range(1, 46):
                o_col = f'o{i}'
                if o_col in row and row[o_col] and int(row[o_col]) > 0:
                    o_values[i] = int(row[o_col])
            if o_values:
                o_values_map[round_num] = o_values

    return pd.DataFrame(rows), o_values_map


def predict(df: pd.DataFrame, target_round: int, o_values_map: dict = None) -> dict:
    """예측 수행 (최근 10회 빈도 기반 개선 알고리즘)

    Args:
        df: 당첨번호 DataFrame
        target_round: 예측할 회차
        o_values_map: {회차: {ord: ball}} o값 매핑 (선택)
    """

    # 예측할 회차의 인덱스 (target_round 전까지의 데이터만 사용)
    df_train = df[df['회차'] < target_round].copy()
    target_idx = len(df_train)

    if target_idx == 0:
        return {
            'round': target_round,
            'top24': [],
            'mid14': [],
            'rest7': [],
            'top24_str': '',
            'mid14_str': '',
            'rest7_str': '',
            'ord_top24_str': '',
            'ord_mid14_str': '',
            'ord_rest7_str': '',
            'status': 'error',
            'message': '학습 데이터 없음'
        }

    # 최근 10회 빈도 분석 (핵심 개선: 전체 빈도 → 최근 10회 빈도)
    recent_n = min(10, len(df_train))
    df_recent = df_train.tail(recent_n)

    recent_freq = {}
    for _, row in df_recent.iterrows():
        for i in range(1, 7):
            ord_col = f'ord{i}'
            if ord_col in row and pd.notna(row[ord_col]) and row[ord_col] > 0:
                ord_val = int(row[ord_col])
                recent_freq[ord_val] = recent_freq.get(ord_val, 0) + 1

    # 1~45 각 번호에 대한 점수 계산 (순수 최근 10회 빈도)
    ord_scores = {}

    for ord_val in range(1, 46):
        # 최근 10회 빈도만 사용 (가장 효과적인 단순 전략)
        score = recent_freq.get(ord_val, 0)
        ord_scores[ord_val] = score

    # 점수 기반 Top24, Mid14, Rest7 추출 (점수순 유지, 오름차순 정렬 제거)
    sorted_scores = sorted(ord_scores.items(), key=lambda x: x[1], reverse=True)
    top24_ords = [ord_val for ord_val, _ in sorted_scores[:24]]
    mid14_ords = [ord_val for ord_val, _ in sorted_scores[24:38]]
    rest7_ords = [ord_val for ord_val, _ in sorted_scores[38:45]]

    # o값 가져오기: 로컬 데이터에서
    o_values = {}
    if o_values_map and target_round in o_values_map:
        o_values = o_values_map[target_round]

    if o_values:
        # ord를 ball로 변환 (o3=15면, ord 3 → ball 15)
        top24_balls = [o_values.get(ord_val, ord_val) for ord_val in top24_ords]
        mid14_balls = [o_values.get(ord_val, ord_val) for ord_val in mid14_ords]
        rest7_balls = [o_values.get(ord_val, ord_val) for ord_val in rest7_ords]
    else:
        # DB 연결 실패 시 ord = ball로 표시
        top24_balls = top24_ords
        mid14_balls = mid14_ords
        rest7_balls = rest7_ords

    # 결과 반환
    result = {
        'round': target_round,
        'top24': top24_balls,
        'mid14': mid14_balls,
        'rest7': rest7_balls,
        'top24_str': ','.join(map(str, top24_balls)),
        'mid14_str': ','.join(map(str, mid14_balls)),
        'rest7_str': ','.join(map(str, rest7_balls)),
        'ord_top24': top24_ords,
        'ord_mid14': mid14_ords,
        'ord_rest7': rest7_ords,
        'ord_top24_str': ','.join(map(str, top24_ords)),
        'ord_mid14_str': ','.join(map(str, mid14_ords)),
        'ord_rest7_str': ','.join(map(str, rest7_ords)),
        'status': 'success',
        'message': f'{target_round}회차 예측 완료 (Top24/Mid14/Rest7)'
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='로또 번호 예측')
    parser.add_argument('--round', '-r', type=int, default=1204,
                        help='예측할 회차 (기본값: 1204)')
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='데이터 파일 경로 (기본값: 1_data/winning_numbers.json)')
    args = parser.parse_args()

    # 데이터 로드
    df, o_values_map = load_data(args.data)

    # 예측할 회차 결정
    target_round = args.round

    print(f"=== {target_round}회차 예측 ===\n")

    # 예측 수행
    result = predict(df, target_round, o_values_map)

    if result['status'] == 'error':
        print(f"오류: {result['message']}")
        return

    # 결과 출력
    print(f"[Ord 번호]")
    print(f"Top24: {result['ord_top24_str']}")
    print(f"Mid14: {result['ord_mid14_str']}")
    print(f"Rest7: {result['ord_rest7_str']}")
    print()
    print(f"[Ball 번호]")
    print(f"Top24: {result['top24_str']}")
    print(f"Mid14: {result['mid14_str']}")
    print(f"Rest7: {result['rest7_str']}")
    print(f"\n{result['message']}")

    # CSV 파일로 저장
    output_path = Path(__file__).parent / f"result_{target_round}.csv"
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['type', 'section', 'numbers'])
        writer.writerow(['ord', 'top24', result['ord_top24_str']])
        writer.writerow(['ord', 'mid14', result['ord_mid14_str']])
        writer.writerow(['ord', 'rest7', result['ord_rest7_str']])
        writer.writerow(['ball', 'top24', result['top24_str']])
        writer.writerow(['ball', 'mid14', result['mid14_str']])
        writer.writerow(['ball', 'rest7', result['rest7_str']])
    print(f"저장됨: {output_path}")


if __name__ == '__main__':
    main()
