# Diabetic-Retinopathy-detect-project_Web-service

당뇨망막병증(DR) 예측 웹서비스 구현 저장소입니다.
프론트/백엔드 연동 파일과 추론 앱을 포함합니다.

## 저장소 범위
- `app.py`: Streamlit 기반 추론 앱
- `best_model_recall.pth`: 학습된 DR 모델 가중치
- `requirements.txt`: Python 의존성
- `DR_model_heatmap_Grad-CAM_map.ipynb`, `DR_Model_Visualization.ipynb`: 시각화 노트북
- `Source_image(input).png`, `Source_image(Output).png`: 예시 이미지

## 요구 사항
- Python 3.10+
- Git LFS (대용량 모델 파일 처리)

## 실행 방법
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

기본 접속 주소: `http://localhost:8501`

## 참고
- `best_model_recall.pth` 파일이 루트 경로에 있어야 합니다.
- 새로 클론한 경우 `git lfs pull`로 대용량 파일을 내려받으세요.
