"""
3단계: 백테스팅

예측 결과와 실제 당첨번호 비교
ord1~ord6이 Top24/Mid14/Rest7 중 어디에 포함되는지 검증
"""
import csv
import json
import argparse
import sys
from pathlib import Path

# predict 모듈 import를 위해 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "2_predict"))
from predict import load_data, predict as run_predict


def load_prediction(round_num: int) -> dict:
    """예측 결과 로드 (CSV 형식)"""
    pred_path = Path(__file__).parent.parent / "2_predict" / f"result_{round_num}.csv"
    result = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            type_key = row['type']
            section = row['section']
            numbers = [int(x) for x in row['numbers'].split(',')]

            if type_key == 'ord':
                result[f'ord_{section}'] = numbers
            else:  # ball
                result[section] = numbers
    return result


def load_actual(round_num: int) -> tuple:
    """실제 당첨번호 로드 (ord1~ord6, ball1~ball6) - CSV 형식

    Returns:
        (list, list): (ord1~ord6, ball1~ball6)
    """
    data_path = Path(__file__).parent.parent / "1_data" / "winning_numbers.csv"
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['round']) == round_num:
                ords = [int(row[f'ord{i}']) for i in range(1, 7)]
                balls = [int(row[f'ball{i}']) for i in range(1, 7)]
                return ords, balls
    return [], []


def backtest(round_num: int) -> dict:
    """백테스트 수행"""
    pred = load_prediction(round_num)
    actual_ords, actual_balls = load_actual(round_num)

    if not actual_ords:
        return {'status': 'error', 'message': f'{round_num}회차 실제 데이터 없음'}

    ord_top24 = set(pred['ord_top24'])
    ord_mid14 = set(pred['ord_mid14'])
    ord_rest7 = set(pred['ord_rest7'])

    # 각 ord가 어느 섹션에 포함되는지 확인
    ord_top_count = 0
    ord_mid_count = 0
    ord_rest_count = 0
    ord_details = []

    for i, (ord_val, ball_val) in enumerate(zip(actual_ords, actual_balls), 1):
        if ord_val in ord_top24:
            section = 'Top24'
            ord_top_count += 1
        elif ord_val in ord_mid14:
            section = 'Mid14'
            ord_mid_count += 1
        elif ord_val in ord_rest7:
            section = 'Rest7'
            ord_rest_count += 1
        else:
            section = '???'

        ord_details.append({'ord': i, 'ord_value': ord_val, 'ball_value': ball_val, 'section': section})

    # 실제 당첨 ord가 예측의 Ball 섹션 중 어디에 있는지 확인
    # 예측 결과에서 ball → (section, ord) 매핑 생성
    ball_top24 = set(pred['top24'])
    ball_mid14 = set(pred['mid14'])
    ball_rest7 = set(pred['rest7'])

    ball_top_count = 0
    ball_mid_count = 0
    ball_rest_count = 0
    ball_details = []

    for i, (ord_val, ball_val) in enumerate(zip(actual_ords, actual_balls), 1):
        # 실제 당첨 ord가 예측의 Ball 섹션 중 어디에 있는지
        if ord_val in ball_top24:
            section = 'Top24'
            ball_top_count += 1
        elif ord_val in ball_mid14:
            section = 'Mid14'
            ball_mid_count += 1
        elif ord_val in ball_rest7:
            section = 'Rest7'
            ball_rest_count += 1
        else:
            section = '???'

        ball_details.append({'ball': i, 'ord_value': ord_val, 'ball_value': ball_val, 'section': section})

    result = {
        'round': round_num,
        'actual_ords': actual_ords,
        'actual_balls': actual_balls,
        'ord_top24_count': ord_top_count,
        'ord_mid14_count': ord_mid_count,
        'ord_rest7_count': ord_rest_count,
        'ord_details': ord_details,
        'ball_top24_count': ball_top_count,
        'ball_mid14_count': ball_mid_count,
        'ball_rest7_count': ball_rest_count,
        'ball_details': ball_details,
        'status': 'success'
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='백테스트')
    parser.add_argument('--round', '-r', type=int, default=1204,
                        help='검증할 회차 (기본값: 1204)')
    args = parser.parse_args()

    result = backtest(args.round)

    if result['status'] == 'error':
        print(f"오류: {result['message']}")
        return

    pred = load_prediction(args.round)

    print(f"=== {result['round']}회차 예측 ===\n")
    print(f"[Ord 번호]")
    print(f"Top24: {','.join(map(str, pred['ord_top24']))}")
    print(f"Mid14: {','.join(map(str, pred['ord_mid14']))}")
    print(f"Rest7: {','.join(map(str, pred['ord_rest7']))}")
    print()
    print(f"[Ball 번호]")
    print(f"Top24: {','.join(map(str, pred['top24']))}")
    print(f"Mid14: {','.join(map(str, pred['mid14']))}")
    print(f"Rest7: {','.join(map(str, pred['rest7']))}")

    print(f"\n=== {result['round']}회차 백테스트 ===\n")
    print(f"실제 당첨 ord:  {result['actual_ords']}")
    print(f"실제 당첨 ball: {result['actual_balls']}\n")

    print("[Ord 섹션별 적중]")
    for d in result['ord_details']:
        print(f"  ord{d['ord']} = {d['ord_value']:2d} → {d['section']}")

    print(f"\n[Ord 요약]")
    print(f"  Top24: {result['ord_top24_count']}개")
    print(f"  Mid14: {result['ord_mid14_count']}개")
    print(f"  Rest7: {result['ord_rest7_count']}개")

    print(f"\n[Ball 섹션별 적중] (실제 당첨 ord가 예측 Ball의 어느 섹션?)")
    for d in result['ball_details']:
        print(f"  ord{d['ball']} = {d['ord_value']:2d} (ball {d['ball_value']:2d}) → {d['section']}")

    print(f"\n[Ball 요약]")
    print(f"  Top24: {result['ball_top24_count']}개")
    print(f"  Mid14: {result['ball_mid14_count']}개")
    print(f"  Rest7: {result['ball_rest7_count']}개")

    # 요약 라인: 회차,ord_top24ord_mid14ord_rest7ball_top24ball_mid14ball_rest7
    summary = f"{result['round']},{result['ord_top24_count']}{result['ord_mid14_count']}{result['ord_rest7_count']}{result['ball_top24_count']}{result['ball_mid14_count']}{result['ball_rest7_count']}"
    print(f"\n{summary}")


def backtest_with_predict(round_num: int, df, o_values_map: dict) -> dict:
    """예측 수행 후 백테스트"""
    # 예측 수행
    pred = run_predict(df, round_num, o_values_map)

    if pred['status'] == 'error':
        return {'status': 'error', 'message': pred['message']}

    # 실제 데이터 로드
    actual_ords, _ = load_actual(round_num)

    if not actual_ords:
        return {'status': 'error', 'message': f'{round_num}회차 실제 데이터 없음'}

    ord_top24 = set(pred['ord_top24'])
    ord_mid14 = set(pred['ord_mid14'])
    ord_rest7 = set(pred['ord_rest7'])

    # Ord 섹션별 카운트
    ord_top_count = 0
    ord_mid_count = 0
    ord_rest_count = 0

    for ord_val in actual_ords:
        if ord_val in ord_top24:
            ord_top_count += 1
        elif ord_val in ord_mid14:
            ord_mid_count += 1
        elif ord_val in ord_rest7:
            ord_rest_count += 1

    # Ball 섹션별 카운트 (실제 당첨 ord가 예측 Ball 섹션 중 어디에?)
    ball_top24 = set(pred['top24'])
    ball_mid14 = set(pred['mid14'])
    ball_rest7 = set(pred['rest7'])

    ball_top_count = 0
    ball_mid_count = 0
    ball_rest_count = 0

    for ord_val in actual_ords:
        if ord_val in ball_top24:
            ball_top_count += 1
        elif ord_val in ball_mid14:
            ball_mid_count += 1
        elif ord_val in ball_rest7:
            ball_rest_count += 1

    return {
        'round': round_num,
        'ord_top24_count': ord_top_count,
        'ord_mid14_count': ord_mid_count,
        'ord_rest7_count': ord_rest_count,
        'ball_top24_count': ball_top_count,
        'ball_mid14_count': ball_mid_count,
        'ball_rest7_count': ball_rest_count,
        'status': 'success'
    }


def run_cumulative_backtest(start_round: int = 826, train_size: int = 50):
    """누적 학습 백테스트 실행

    Args:
        start_round: 시작 회차 (기본값: 826)
        train_size: 초기 학습 데이터 수 (기본값: 50)
    """
    # 데이터 로드
    df, o_values_map = load_data()

    # 전체 회차 목록
    all_rounds = sorted(df['회차'].unique())

    # 시작 인덱스 찾기
    start_idx = all_rounds.index(start_round) if start_round in all_rounds else 0

    # 첫 예측 회차 = start_round + train_size
    first_predict_idx = start_idx + train_size

    if first_predict_idx >= len(all_rounds):
        print(f"오류: 학습 데이터가 부족합니다. (전체 {len(all_rounds)}회차, 필요 {train_size + 1}회차)")
        return

    # CSV 출력 경로
    output_path = Path(__file__).parent / "backtest_results.csv"

    results = []

    print(f"누적 백테스트 시작: {start_round}회차부터 {train_size}회차 학습")
    print(f"예측 시작: {all_rounds[first_predict_idx]}회차")
    print()

    for i in range(first_predict_idx, len(all_rounds)):
        target_round = all_rounds[i]
        train_count = i - start_idx  # 학습에 사용된 회차 수

        result = backtest_with_predict(target_round, df, o_values_map)

        if result['status'] == 'success':
            summary_code = f"{result['ord_top24_count']}{result['ord_mid14_count']}{result['ord_rest7_count']}{result['ball_top24_count']}{result['ball_mid14_count']}{result['ball_rest7_count']}"
            results.append({
                'round': target_round,
                'train_count': train_count,
                'summary': summary_code,
                'ord_top24': result['ord_top24_count'],
                'ord_mid14': result['ord_mid14_count'],
                'ord_rest7': result['ord_rest7_count'],
                'ball_top24': result['ball_top24_count'],
                'ball_mid14': result['ball_mid14_count'],
                'ball_rest7': result['ball_rest7_count']
            })
            print(f"{target_round},{summary_code} (학습: {train_count}회차)")
        else:
            print(f"{target_round}: {result['message']}")

    # CSV 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("round,train_count,summary,ord_top24,ord_mid14,ord_rest7,ball_top24,ball_mid14,ball_rest7\n")
        for r in results:
            f.write(f"{r['round']},{r['train_count']},{r['summary']},{r['ord_top24']},{r['ord_mid14']},{r['ord_rest7']},{r['ball_top24']},{r['ball_mid14']},{r['ball_rest7']}\n")

    print(f"\n저장 완료: {output_path}")
    print(f"총 {len(results)}회차 백테스트 완료")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='백테스트')
    parser.add_argument('--round', '-r', type=int, default=None,
                        help='검증할 회차 (미지정시 누적 백테스트 실행)')
    parser.add_argument('--cumulative', '-c', action='store_true',
                        help='누적 백테스트 실행')
    parser.add_argument('--start', '-s', type=int, default=826,
                        help='시작 회차 (기본값: 826)')
    parser.add_argument('--train', '-t', type=int, default=50,
                        help='초기 학습 회차 수 (기본값: 50)')
    args = parser.parse_args()

    if args.cumulative or args.round is None:
        run_cumulative_backtest(args.start, args.train)
    else:
        main()
