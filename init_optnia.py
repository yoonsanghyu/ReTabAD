import optuna

# 필요한 테이블을 자동으로 생성
optuna.create_study(
    study_name="init",  # 아무 이름으로 생성
    storage="sqlite:///exp_db/retab.sqlite3",
    load_if_exists=True
)
