import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import io

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="EC값에 따른 상하부 길이의 성장률 차이",
    layout="wide"
)

# 한글 폰트 깨짐 방지 (Streamlit)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 유틸: NFC/NFD 안전 파일 탐색
# =========================================================
def normalize_name(name: str) -> str:
    return unicodedata.normalize("NFC", name)

# =========================================================
# 데이터 로딩
# =========================================================
@st.cache_data
def load_environment_data():
    data_dir = Path("data")
    if not data_dir.exists():
        st.error("❌ data 폴더를 찾을 수 없습니다.")
        return None

    env_data = {}

    for file in data_dir.iterdir():
        if file.suffix.lower() == ".csv":
            normalized = normalize_name(file.name)
            if "환경데이터" in normalized:
                school = normalized.replace("_환경데이터.csv", "")
                env_data[school] = pd.read_csv(file)

    if not env_data:
        st.error("❌ 환경 데이터 CSV 파일을 찾을 수 없습니다.")
        return None

    return env_data


@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    xlsx_file = None

    for file in data_dir.iterdir():
        if file.suffix.lower() == ".xlsx":
            if "생육결과" in normalize_name(file.name):
                xlsx_file = file
                break

    if xlsx_file is None:
        st.error("❌ 생육결과 XLSX 파일을 찾을 수 없습니다.")
        return None

    # 시트명 하드코딩 금지
    growth_data = pd.read_excel(xlsx_file, sheet_name=None)
    return growth_data


with st.spinner("📂 데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

if env_data is None or growth_data is None:
    st.stop()

# =========================================================
# 기본 정보
# =========================================================
EC_MAP = {
    "송도고": 1.0,
    "하늘고": 2.0,
    "아라고": 4.0,
    "동산고": 8.0
}

schools = ["전체"] + sorted(env_data.keys())

# =========================================================
# 사이드바
# =========================================================
selected_school = st.sidebar.selectbox("🏫 학교 선택", schools)

# =========================================================
# 제목
# =========================================================
st.title("🌱 EC값에 따른 상하부 길이의 성장률 차이")

tabs = st.tabs([
    "📊 학교별 평균 환경데이터 & 이탈행",
    "📈 EC값에 따른 성장량 비교",
    "🔬 지상부 vs 지하부 관계"
])

# =========================================================
# TAB 1
# =========================================================
with tabs[0]:
    st.subheader("학교별 평균 환경 데이터 분석")

    rows = []
    outliers = []

    for school, df in env_data.items():
        rows.append({
            "학교": school,
            "평균 온도": df["temperature"].mean(),
            "평균 습도": df["humidity"].mean(),
            "평균 pH": df["ph"].mean(),
            "평균 EC": df["ec"].mean()
        })

        # 이탈행(EC 기준 ±30%)
        target = EC_MAP.get(school, None)
        if target is not None:
            mask = (df["ec"] < target * 0.7) | (df["ec"] > target * 1.3)
            if mask.any():
                temp = df.loc[mask].copy()
                temp["학교"] = school
                outliers.append(temp)

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df)

    if outliers:
        st.markdown("### ⚠️ EC 이탈 측정값")
        outlier_df = pd.concat(outliers)
        st.dataframe(outlier_df)

        buffer = io.BytesIO()
        outlier_df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            label="📥 이탈행 데이터 다운로드",
            data=buffer,
            file_name="EC_이탈행_목록.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================================================
# TAB 2
# =========================================================
with tabs[1]:
    st.subheader("EC값에 따른 성장량 비교")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["지상부 길이 평균", "지하부 길이 평균"]
    )

    for school, df in growth_data.items():
        ec = EC_MAP.get(school, None)
        if ec is None:
            continue

        fig.add_trace(
            go.Bar(
                x=[ec],
                y=[df["지상부 길이(mm)"].mean()],
                name=f"{school} 지상부"
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=[ec],
                y=[df["지하부길이(mm)"].mean()],
                name=f"{school} 지하부"
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=500,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3
# =========================================================
with tabs[2]:
    st.subheader("EC값에 따른 지상부–지하부 관계")

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["지상부 길이 vs 지하부 길이"]
    )

    for school, df in growth_data.items():
        ec = EC_MAP.get(school, None)
        if ec is None:
            continue

        fig.add_trace(
            go.Scatter(
                x=df["지상부 길이(mm)"],
                y=df["지하부길이(mm)"],
                mode="markers",
                name=f"{school} (EC {ec})"
            )
        )

    fig.update_layout(
        xaxis_title="지상부 길이 (mm)",
        yaxis_title="지하부 길이 (mm)",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)
