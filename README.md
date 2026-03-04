# Diabetic-Retinopathy-detect-project_Web-service

당뇨망막병증 예측 서비스 구현 저장소입니다.

## 개요
추론 앱 실행 코드와 모델 파일을 포함하며, DR 예측 결과를 웹 인터페이스로 제공합니다.

## 주요 파일
- `app.py`: Streamlit 앱 엔트리포인트
- `best_model_recall.pth`: 추론용 모델 가중치
- `requirements.txt`: 실행 의존성
- 시각화 노트북: Grad-CAM 및 모델 결과 해석

## 실행 방법
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

기본 접속 주소: `http://localhost:8501`

## 비고
대용량 모델 파일 사용 시 Git LFS 환경이 필요합니다.
