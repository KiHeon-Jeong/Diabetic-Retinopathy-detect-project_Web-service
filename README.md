# DR Web-service Module (Frontend/Backend Integration)

## 1) 개요
본 구현 코드는 프로젝트 DR 서비스 파트입니다.  
학습된 DR 모델을 기반으로 Streamlit 추론 앱을 구성하고, 사용자 입력 이미지에 대한 예측 결과를 제공하는 흐름으로 구성되어 있습니다.

## 2) 파일 구성
1. `app.py`  
   DR 추론 Streamlit 애플리케이션
2. `best_model_recall.pth`  
   서비스 추론용 학습 가중치 파일
3. `requirements.txt`  
   실행 의존성 목록
4. `DR_model_heatmap_Grad-CAM_map.ipynb`  
   Grad-CAM 기반 시각화 노트북
5. `DR_Model_Visualization.ipynb`  
   모델 결과 해석/시각화 노트북
6. `model_1_RN50_DS_1_ver4.ipynb`  
   모델 실험 보조 노트북
7. `Source_image(input).png`, `Source_image(Output).png`  
   입력/출력 예시 이미지

## 3) 서비스 파이프라인 요약
- 사용자 이미지 업로드
- 전처리(Resize/Normalize)
- DR 분류 모델 추론
- 예측 확률 및 결과 화면 출력
- Grad-CAM 기반 해석 정보 제공

## 4) 실행 메모
- 실행 명령:
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
- 기본 접속 주소: `http://localhost:8501`
- 대용량 모델 파일은 Git LFS 환경에서 관리됩니다.

## 5) 수행 내용
- 모델 추론 로직 서비스화
- 시각화 기반 결과 해석 기능 반영
- UI 흐름 기반 입력-결과 연동 구현

## 6) 기술 스택
- Python
- Streamlit
- PyTorch / torchvision
- Pillow

## 7) 참고
- EDA/모델링 실험 코드는 `Diabetic-Retinopathy-detect-project` 저장소에서 관리합니다.
