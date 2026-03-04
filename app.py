# 표준 라이브러리
import io
from pathlib import Path

# 서드파티 라이브러리
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 경로 및 모델 파일
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "best_model_recall.pth"
# 판단 근거 이미지 경로
REASON_IMAGE_PATH = APP_DIR / "151_visualization.png"

# 전처리 상수(학습과 동일)
# - IMAGE_SIZE: 입력 리사이즈 크기
# - MEAN/STD: ImageNet 정규화 값
IMAGE_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# 모델 헤드에 사용되는 어텐션 블록
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 채널 어텐션 맵 생성 후 입력에 곱하는 구조
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 어텐션 가중치 계산
        att = self.attention(x)
        # 입력 특징맵에 어텐션 적용
        return x * att


# ResNet50 백본 + 어텐션 + 이진 분류기
class ResNet50AttentionBinaryDR(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # ResNet-50 백본 로딩
        resnet = models.resnet50(pretrained=pretrained)
        # 마지막 FC 이전의 특징 추출기만 사용
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # 어텐션 모듈
        self.attention = AttentionModule(2048)
        # 전역 평균 풀링
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 이진 분류기 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        # 특징 추출
        x = self.features(x)
        # 어텐션 적용
        x = self.attention(x)
        # 전역 평균 풀링
        x = self.gap(x)
        # 2D -> 1D flatten
        x = x.view(x.size(0), -1)
        # 분류기 통과
        x = self.classifier(x)
        return x


# 이미지 전처리 파이프라인
def get_preprocess():
    # 학습 시 사용한 전처리와 동일하게 구성
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


# 매 인터랙션마다 재로딩하지 않도록 모델 캐시
@st.cache_resource
def load_model(model_path: Path, device: torch.device):
    # 모델 구조 생성
    model = ResNet50AttentionBinaryDR(pretrained=False)
    # 체크포인트 로딩 (dict 또는 state_dict 자체)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    # 디바이스 이동 및 평가 모드
    model.to(device)
    model.eval()
    return model


# 추론 실행 후 0~1 확률 반환
def predict(image: Image.Image, model: nn.Module, device: torch.device):
    # 단일 이미지 전처리 후 배치 차원 추가
    preprocess = get_preprocess()
    tensor = preprocess(image).unsqueeze(0).to(device)
    # 추론은 그래디언트 비활성화
    with torch.no_grad():
        logits = model(tensor)
        # 시그모이드로 확률 변환
        prob = torch.sigmoid(logits).item()
    return prob


# 세션 상태에 이미지 메타데이터 저장
def set_image_state(image_bytes, name, source):
    # 업로드한 이미지 정보를 세션에 저장
    st.session_state["image_bytes"] = image_bytes
    st.session_state["image_name"] = name
    st.session_state["image_source"] = source


# CSS 카드 래퍼용 앵커 마커
def card_anchor(anchor_id):
    # CSS에서 카드 블록을 감싸기 위한 숨김 앵커
    st.markdown(f'<div class="card-anchor" id="{anchor_id}"></div>', unsafe_allow_html=True)


# 위험도 바(HTML/CSS) 렌더링
def render_bar(prob):
    # 확률을 0~1 범위로 클램핑
    value = max(0.0, min(1.0, prob))
    # 퍼센트 정수화
    percent = int(round(value * 100))
    return f"""
    <div class="risk-bar">
        <div class="risk-fill" style="width: {percent}%"></div>
    </div>
    <div class="risk-bar-labels">
        <span>낮음</span>
        <span>높음</span>
    </div>
    """


# 페이지 설정
st.set_page_config(page_title="DR Binary Classifier", layout="wide")

# 타이포/레이아웃 커스텀 스타일
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Noto+Sans+KR:wght@300;400;600;700&display=swap');

    /* 색상 토큰 */
    :root {
        --bg: #f3f5eb;
        --ink: #1f2a17;
        --muted: #5b634f;
        --sidebar: #3e4b10;
        --sidebar-light: #4a5b15;
        --accent: #bcd874;
        --accent-2: #88a63f;
        --card: #f7f8ef;
        --border: #e2e6d4;
        --shadow: 0 10px 26px rgba(31, 42, 23, 0.08);
    }

    .stApp {
        background: var(--bg);
        color: var(--ink);
        font-family: "Noto Sans KR", sans-serif;
    }

    /* 타이틀 폰트 */
    h1, h2, h3 {
        font-family: "Gowun Batang", serif;
        color: var(--ink);
    }

    /* 좌측 패널 스타일 */
    div[data-testid="column"]:has(#left-panel-marker) {
        background: linear-gradient(180deg, var(--sidebar), #33400b);
        color: #f0f3d9;
        border-radius: 22px;
        padding: 26px 24px;
        min-height: 720px;
        box-shadow: var(--shadow);
    }

    .left-title {
        font-size: 70px;
        line-height: 1.35;
        margin: 0 0 8px 0;
        font-family: "Gowun Batang", serif;
    }

    /* 좌측 서브 텍스트 */
    .left-sub {
        color: #000000;
        font-size: 25px;
        margin: 0 0 20px 0;
    }

    /* 좌측 리스트 */
    .left-list {
        margin: 16px 0 20px 0;
        padding-left: 14px;
        font-size: 15px;
        color: #000000;
    }

    /* 기본 카드 스타일 */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        color: var(--ink);
    }

    .card-anchor {
        display: none;
    }

    /* Streamlit 블록을 카드처럼 보이게 래핑 */
    div[data-testid="stVerticalBlock"]:has(.card-anchor) {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: var(--shadow);
        margin-bottom: 16px;
        color: var(--ink);
    }

    div[data-testid="column"]:has(#left-panel-marker) div[data-testid="stVerticalBlock"]:has(.card-anchor) {
        background: #f7f8ef;
    }

    /* 카드 제목 */
    .card-title {
        font-size: 25px;
        margin-bottom: 10px;
        font-weight: 700;
    }

    /* 퍼센트 숫자 */
    .metric {
        font-size: 64px;
        font-weight: 700;
        color: #3e4b10;
        margin: 0 0 6px 0;
    }

    /* 메트릭 서브 라벨 */
    .metric-label {
        font-size: 13px;
        color: var(--muted);
        margin-top: 2px;
    }

    /* 위험도 바 */
    .risk-bar {
        position: relative;
        width: 100%;
        height: 14px;
        background: #e3e5d8;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 10px;
    }

    .risk-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #7aa83f, #f0c04b 55%, #d66a2c);
        transition: width 0.4s ease;
    }

    /* 위험도 바 라벨 */
    .risk-bar-labels {
        display: flex;
        justify-content: space-between;
        font-size: 16px;
        color: var(--muted);
        margin-top: 6px;
    }

    /* 위험도 배지 */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 13px;
    }

    .badge-dr {
        background: rgba(201, 92, 38, 0.16);
        color: #7a3c1b;
        border: 1px solid rgba(201, 92, 38, 0.25);
    }

    .badge-nodr {
        background: rgba(72, 124, 69, 0.12);
        color: #2f5c2b;
        border: 1px solid rgba(72, 124, 69, 0.18);
    }

    /* 업로더 라벨 */
    .stFileUploader label {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 이미지 페이로드용 세션 상태 초기화
if "image_bytes" not in st.session_state:
    # 업로드한 파일 바이트
    st.session_state["image_bytes"] = None
if "image_name" not in st.session_state:
    # 업로드한 파일명
    st.session_state["image_name"] = None
if "image_source" not in st.session_state:
    # 입력 소스(현재는 manual만 사용)
    st.session_state["image_source"] = None

# 메인 레이아웃(좌측 패널, 우측 컨텐츠)
left, right = st.columns([0.34, 0.66], gap="large")
image = None

with left:
    # 좌측 패널 헤더
    st.markdown('<div id="left-panel-marker"></div>', unsafe_allow_html=True)
    st.markdown('<div class="left-title">당뇨병성 망막병증<br/>예측 모델</div>', unsafe_allow_html=True)
    st.markdown('<div class="left-sub">조기 선별이 시력을 지킵니다</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="left-list">
            <li>안저 이미지를 업로드하고 예측을 시작하세요.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        # 입력 요약 카드 + 업로더
        card_anchor("card-input-summary")
        st.markdown('<div class="card-title">입력 요약</div>', unsafe_allow_html=True)
        # 파일 업로드 위젯
        uploaded = st.file_uploader(
            "안저 이미지를 선택하세요",
            type=["png", "jpg", "jpeg"],
            label_visibility="visible",
            key="manual_uploader",
        )
        if uploaded:
            image_bytes = uploaded.read()
            # 업로드된 이미지를 세션에 저장
            set_image_state(image_bytes, uploaded.name, "manual")

        # 메타데이터 표시
        if st.session_state["image_bytes"] is not None:
            # 이미지 로딩
            image = Image.open(io.BytesIO(st.session_state["image_bytes"])).convert("RGB")
            st.write(f"파일명: {st.session_state['image_name']}")
            st.write(f"해상도: {image.size[0]} x {image.size[1]} px")
            st.write("형식: RGB")
            st.write(f"파일 크기: {len(st.session_state['image_bytes']) / (1024 ** 2):.2f} MB")
        else:
            st.write("데이터를 불러오면 요약이 표시됩니다.")

    with st.container():
        # 입력 이미지 카드(입력 요약 아래)
        card_anchor("card-input-image-left")
        st.markdown('<div class="card-title">입력 이미지</div>', unsafe_allow_html=True)
        if image is not None:
            # 업로드된 이미지 표시
            st.image(image, caption="입력 이미지", use_container_width=True)
        else:
            st.write("이미지 미리보기가 여기에 표시됩니다.")

with right:
    # 모델 로딩 및 추론 준비
    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        # 모델 파일 없으면 에러 표시 후 중단
        st.error(f"Model file not found: {model_path}")
        st.stop()

    # 가능한 경우 GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    # 추론 결과 변수 초기화
    prob = None
    is_dr = None
    label = None

    # 세션 상태에서 이미지 확보
    if image is None and st.session_state["image_bytes"] is not None:
        # 좌측에서 업로드된 이미지를 우측에서도 사용
        image = Image.open(io.BytesIO(st.session_state["image_bytes"])).convert("RGB")

    # 확률 계산 및 임계값 판정
    if image is not None:
        # 모델 추론
        prob = predict(image, model, device)
        # 임계값 기준 이진 분류
        threshold = 0.5
        is_dr = prob >= threshold
        label = "DR(당뇨병성 망막병증) 의심 됨" if is_dr else "정상 범위"

    with st.container():
        # 예측 결과 카드(수치 + 바)
        card_anchor("card-prediction")
        st.markdown('<div class="card-title">예측 결과</div>', unsafe_allow_html=True)
        if prob is not None:
            # 위험도 배지 계산
            badge_class = "badge-dr" if is_dr else "badge-nodr"
            risk_label = "높은 위험도" if prob >= 0.7 else "중간 위험도" if prob >= 0.4 else "낮은 위험도"
            st.markdown(
                f"""
            <p class="metric" style="font-size: 72px;">{prob * 100:.0f}%</p>
                <div class="metric-label">DR 확률</div>
                <div style="margin: 10px 0 8px 0;">
                    <span class="badge {badge_class}">{risk_label}</span>
                </div>
                <div style="font-size: 25px; color: var(--muted);">예측: {label}</div>
                """,
                unsafe_allow_html=True,
            )
            # 위험도 바 그래프 표시
            st.markdown(render_bar(prob), unsafe_allow_html=True)
        else:
            st.info("이미지를 업로드하면 예측 결과를 확인할 수 있습니다.")

    with st.container():
        # 판단 근거 카드
        card_anchor("card-evidence")
        st.markdown('<div class="card-title">판단 근거</div>', unsafe_allow_html=True)
        if image is not None and REASON_IMAGE_PATH.exists():
            # 입력 이미지가 있을 때만 판단 근거 표시
            st.image(str(REASON_IMAGE_PATH), caption="판단 근거", width=1400)
        elif image is not None:
            st.write("판단 근거 이미지를 찾을 수 없습니다.")
        else:
            st.write("입력 이미지가 업로드되면 판단 근거가 표시됩니다.")

    with st.container():
        # 권장 조치 카드
        card_anchor("card-actions")
        st.markdown('<div class="card-title">권장 조치</div>', unsafe_allow_html=True)
        if prob is not None:
            # 결과에 따른 조치 안내
            if is_dr:
                st.write(
                    "- 전문의 진료를 예약하고 산동 안저검사로 확인하세요.\n"
                    "- 임상적으로 필요하면 OCT/FA를 고려하세요.\n"
                    "- 과거 이미지 및 증상과 비교하세요."
                )
            else:
                st.write(
                    "- 가이드라인에 따른 정기 검진을 유지하세요.\n"
                    "- 증상이 있으면 결과와 무관하게 진료를 받으세요.\n"
                    "- 이미지 품질이 낮으면 재촬영을 권장합니다."
                )
        else:
            st.write("예측 결과에 따라 임상적 조치를 안내합니다.")

    with st.container():
        # 모델 메타데이터 카드
        card_anchor("card-model-info")
        st.markdown('<div class="card-title">모델 정보</div>', unsafe_allow_html=True)
        st.write("모델: ResNet-50 + Attention")
        st.write(f"디바이스: {device.type.upper()}")
        st.write("결정 임계값: 0.50") 
