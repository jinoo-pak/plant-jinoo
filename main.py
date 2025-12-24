# main.py
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ------------------------------ 기본 설정 ------------------------------ #
st.set_page_config(
    page_title="EC값에 따른 상하부 길이의 성장률 차이",
    layout="wide",
)

# 한글 폰트 설정 (웹 + Plotly)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


def apply_korean_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig


# ------------------------------ 상수 / 메타 정보 ------------------------------ #
SCHOOL_INFO = {
    "송도고": {"display": "송도고", "ec": 1.0, "color": "#1f77b4"},
    "하늘고": {"display": "하늘고", "ec": 2.0, "color": "#2ca02c"},
    "아라고": {"display": "아라고", "ec": 4.0, "color": "#ff7f0e"},
    "동산고": {"display": "동산고", "ec": 8.0, "color": "#9467bd"},
}

EC_TO_SCHOOL = {}
for _school_name, _info in SCHOOL_INFO.items():
    EC_TO_SCHOOL[_info["ec"]] = _school_name

COL_PLANT_ID = "개체번호"
COL_LEAF = "잎 수(장)"
COL_SHOOT = "지상부 길이(mm)"
COL_ROOT = "지하부길이(mm)"
COL_FRESH = "생중량(g)"


# ------------------------------ 유틸 함수 ------------------------------ #
def contains_normalized(haystack: str, needle: str) -> bool:
    """NFC/NFD 차이를 고려한 부분 문자열 검색."""
    if haystack is None or needle is None:
        return False
    for form in ("NFC", "NFD"):
        h = unicodedata.normalize(form, haystack)
        n = unicodedata.normalize(form, needle)
        if n in h:
            return True
    return False


@st.cache_data
def find_data_paths(data_dir_str: str):
    """data/ 폴더에서 환경 CSV와 생육 결과 Excel 경로 탐색."""
    base_dir = Path(data_dir_str)
    env_paths = {}
    growth_excel_path = None

    if not base_dir.exists():
        return env_paths, growth_excel_path

    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name

        suffix = path.suffix.lower()
        # 환경 데이터 CSV
        if suffix == ".csv":
            if not contains_normalized(name, "환경데이터"):
                continue
            matched_school = None
            for school_name in SCHOOL_INFO.keys():
                if contains_normalized(name, school_name):
                    matched_school = school_name
                    break
            if matched_school is not None:
                env_paths[matched_school] = str(path)

        # 생육 결과 Excel
        elif suffix in (".xlsx", ".xls"):
            if contains_normalized(name, "생육결과데이터"):
                growth_excel_path = str(path)

    return env_paths, growth_excel_path


@st.cache_data
def load_environment_data(env_path_map: dict):
    """학교별 환경 데이터 로딩 및 통합."""
    env_by_school = {}
    frames = []

    for school_name, path_str in env_path_map.items():
        csv_path = Path(path_str)
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.sort_values("time")

        df["학교"] = school_name
        env_by_school[school_name] = df
        frames.append(df)

    if frames:
        env_all = pd.concat(frames, ignore_index=True)
    else:
        env_all = pd.DataFrame()

    return env_by_school, env_all


@st.cache_data
def load_growth_data(excel_path_str: str):
    """생육 결과 Excel에서 모든 시트를 읽어 학교별 데이터로 통합."""
    excel_path = Path(excel_path_str)
    if not excel_path.exists():
        return None, {}

    sheets_dict = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

    growth_frames = []
    growth_by_school = {}

    for sheet_name, df in sheets_dict.items():
        matched_school = None
        for school_name in SCHOOL_INFO.keys():
            if contains_normalized(sheet_name, school_name):
                matched_school = school_name
                break

        if matched_school is None:
            continue

        df_local = df.copy()

        if COL_PLANT_ID in df_local.columns:
            df_local = df_local.dropna(
                subset=[COL_PLANT_ID, COL_FRESH], how="all"
            )

        df_local["학교"] = matched_school
        df_local["EC"] = SCHOOL_INFO[matched_school]["ec"]

        numeric_cols = [COL_LEAF, COL_SHOOT, COL_ROOT, COL_FRESH]
        for col in numeric_cols:
            if col in df_local.columns:
                df_local[col] = pd.to_numeric(df_local[col], errors="coerce")

        growth_frames.append(df_local)
        growth_by_school[matched_school] = df_local

    if not growth_frames:
        return None, {}

    growth_all = pd.concat(growth_frames, ignore_index=True)
    return growth_all, growth_by_school


def make_env_summary(env_all: pd.DataFrame) -> pd.DataFrame:
    """학교별 평균 환경 요약."""
    if env_all is None or env_all.empty:
        return pd.DataFrame()

    summary = (
        env_all.groupby("학교")
        .agg(
            측정횟수=("time", "count"),
            평균온도=("temperature", "mean"),
            평균습도=("humidity", "mean"),
            평균pH=("ph", "mean"),
            평균EC=("ec", "mean"),
        )
        .reset_index()
    )

    summary["목표EC"] = summary["학교"].map(
        lambda x: SCHOOL_INFO.get(x, {}).get("ec")
    )
    summary["색상"] = summary["학교"].map(
        lambda x: SCHOOL_INFO.get(x, {}).get("color")
    )

    summary = summary.sort_values("목표EC")
    return summary


def compute_env_outliers(env_all: pd.DataFrame) -> pd.DataFrame:
    """학교별 평균 대비 3표준편차 이상 이탈한 환경 데이터 행 추출."""
    if env_all is None or env_all.empty:
        return pd.DataFrame()

    df = env_all.copy()
    metrics = ["temperature", "humidity", "ph", "ec"]
    for col in metrics:
        if col not in df.columns:
            df[col] = np.nan

    for col in metrics:
        mean_col = df.groupby("학교")[col].transform("mean")
        std_col = df.groupby("학교")[col].transform("std")
        diff = (df[col] - mean_col).abs()
        is_out = (std_col > 0) & (diff > 3 * std_col)
        df["이탈_" + col] = is_out

    cond = False
    for col in metrics:
        cond = cond | df["이탈_" + col]

    out_df = df[cond].copy()
    if out_df.empty:
        return pd.DataFrame()

    def make_reason(row):
        items = []
        name_map = {
            "temperature": "온도",
            "humidity": "습도",
            "ph": "pH",
            "ec": "EC",
        }
        for col in metrics:
            if bool(row.get("이탈_" + col, False)):
                items.append(name_map.get(col, col))
        return ", ".join(items)

    out_df["이탈_항목"] = out_df.apply(make_reason, axis=1)

    keep_cols = [
        "학교",
        "time",
        "temperature",
        "humidity",
        "ph",
        "ec",
        "이탈_항목",
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]
    out_df = out_df[keep_cols].sort_values(["학교", "time"])

    return out_df


def make_ec_growth_summary(growth_df: pd.DataFrame) -> pd.DataFrame:
    """EC별 성장 및 생중량 요약."""
    if growth_df is None or growth_df.empty:
        return pd.DataFrame()

    summary = (
        growth_df.groupby("EC")
        .agg(
            평균지상부길이=(COL_SHOOT, "mean"),
            평균지하부길이=(COL_ROOT, "mean"),
            평균생중량=(COL_FRESH, "mean"),
            개체수=(COL_PLANT_ID, "count"),
        )
        .reset_index()
    )
    summary = summary.sort_values("EC")
    summary["학교"] = summary["EC"].map(lambda x: EC_TO_SCHOOL.get(x))
    summary["색상"] = summary["학교"].map(
        lambda x: SCHOOL_INFO.get(x, {}).get("color")
    )
    return summary


def format_float(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    try:
        return str(round(float(x), digits))
    except Exception:
        return "-"


# ------------------------------ 메인 앱 ------------------------------ #
def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    st.title("EC값에 따른 상하부 길이의 성장률 차이")

    # 데이터 경로 탐색
    with st.spinner("데이터 파일 경로를 탐색하는 중입니다..."):
        env_paths, growth_excel_path = find_data_paths(str(data_dir))

    if not env_paths:
        st.error("data 폴더에서 환경 데이터 CSV 파일(이름에 '환경데이터' 포함)을 찾을 수 없습니다.")
    if growth_excel_path is None:
        st.error("data 폴더에서 생육 결과 Excel 파일(이름에 '생육결과데이터' 포함)을 찾을 수 없습니다.")

    if not env_paths or growth_excel_path is None:
        st.stop()

    # 데이터 로딩
    with st.spinner("환경 데이터 로딩 및 전처리 중입니다..."):
        env_by_school, env_all = load_environment_data(env_paths)

    with st.spinner("생육 결과 데이터 로딩 및 전처리 중입니다..."):
        growth_all, growth_by_school = load_growth_data(growth_excel_path)

    if env_all is None or env_all.empty:
        st.error("환경 데이터에 유효한 레코드가 없습니다.")
        st.stop()

    if growth_all is None or growth_all.empty:
        st.error("생육 결과 데이터에 유효한 레코드가 없습니다.")
        st.stop()

    # 기본 요약 계산
    env_summary = make_env_summary(env_all)
    env_outliers = compute_env_outliers(env_all)
    ec_growth_summary_all = make_ec_growth_summary(growth_all)

    optimal_ec = None
    optimal_weight = None
    optimal_school = None
    if not ec_growth_summary_all.empty:
        idx_opt = ec_growth_summary_all["평균생중량"].idxmax()
        row_opt = ec_growth_summary_all.loc[idx_opt]
        optimal_ec = float(row_opt["EC"])
        optimal_weight = float(row_opt["평균생중량"])
        optimal_school = EC_TO_SCHOOL.get(optimal_ec)

    # 사이드바: 학교 선택
    school_options = ["전체"] + list(SCHOOL_INFO.keys())
    selected_school = st.sidebar.selectbox("학교 선택", options=school_options, index=0)

    if selected_school == "전체":
        env_selected = env_all
        growth_selected = growth_all
    else:
        env_selected = env_all[env_all["학교"] == selected_school]
        growth_selected = growth_all[growth_all["학교"] == selected_school]

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(
        [
            "탭1. 학교별 평균 환경데이터 & 이탈행",
            "탭2. EC값에 따른 성장량 (학교별 비교)",
            "탭3. EC값에 따른 지상부·지하부 길이 관계",
        ]
    )

    # ====================== 탭 1 ====================== #
    with tab1:
        st.subheader("학교별 평균 환경데이터 분석")

        if env_summary.empty:
            st.error("환경 평균 데이터를 계산할 수 없습니다.")
        else:
            st.markdown(
                "- 각 학교의 센서 데이터를 이용해 **평균 온도, 평균 습도, 평균 pH, 평균 EC**를 산출했습니다.\n"
                "- 목표 EC 값(실험 설계)과 실측 평균 EC 값을 함께 비교할 수 있습니다."
            )
            st.dataframe(env_summary, hide_index=True)

        st.markdown("---")
        st.subheader("환경 데이터 이탈행 목록 (3σ 기준)")

        if env_outliers.empty:
            st.info("3표준편차 기준으로 판정된 이탈행이 없습니다.")
        else:
            if selected_school == "전체":
                out_sel = env_outliers
            else:
                out_sel = env_outliers[env_outliers["학교"] == selected_school]

            if out_sel.empty:
                st.info("선택한 학교에서 3표준편차 기준 이탈행이 없습니다.")
            else:
                st.markdown(
                    "- 각 행은 해당 학교의 평균 대비 **3표준편차를 초과하는 항목**이 하나 이상 있을 때 이탈행으로 표시됩니다.\n"
                    "- `이탈_항목` 컬럼에 어떤 항목(온도/습도/pH/EC)이 기준을 벗어났는지 정리했습니다."
                )
                st.dataframe(out_sel, hide_index=True)

                # XLSX 다운로드 (BytesIO 필수)
                buffer = io.BytesIO()
                out_sel.to_excel(buffer, index=False, engine="openpyxl")
                buffer.seek(0)

                if selected_school == "전체":
                    file_name = "환경데이터_이탈행_전체.xlsx"
                else:
                    file_name = "환경데이터_이탈행_" + selected_school + ".xlsx"

                st.download_button(
                    label="이탈행 목록 Excel 다운로드",
                    data=buffer,
                    file_name=file_name,
                    mime=(
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    ),
                )

    # ====================== 탭 2 ====================== #
    with tab2:
        st.subheader("EC값에 따른 성장량 (전체 학교 기준)")

        if ec_growth_summary_all.empty:
            st.error("EC별 성장량 요약을 계산할 수 없습니다.")
        else:
            fig_growth = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "EC별 평균 지상부 길이 (mm)",
                    "EC별 평균 지하부 길이 (mm)",
                    "EC별 평균 생중량 (g)",
                ),
                horizontal_spacing=0.08,
            )

            ec_values = list(ec_growth_summary_all["EC"])
            x_labels = [str(v) for v in ec_values]
            colors_by_ec = []
            for ec_val in ec_values:
                sname = EC_TO_SCHOOL.get(ec_val)
                base_color = "#888888"
                if sname in SCHOOL_INFO:
                    base_color = SCHOOL_INFO[sname]["color"]
                colors_by_ec.append(base_color)

            # 1) 지상부 길이
            fig_growth.add_trace(
                go.Bar(
                    x=x_labels,
                    y=ec_growth_summary_all["평균지상부길이"],
                    marker_color=colors_by_ec,
                    name="평균 지상부 길이",
                ),
                row=1,
                col=1,
            )

            # 2) 지하부 길이
            fig_growth.add_trace(
                go.Bar(
                    x=x_labels,
                    y=ec_growth_summary_all["평균지하부길이"],
                    marker_color=colors_by_ec,
                    name="평균 지하부 길이",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # 3) 생중량 (하늘고 EC 2.0 강조)
            weight_colors = []
            for ec_val, base_color in zip(ec_values, colors_by_ec):
                if optimal_ec is not None and abs(ec_val - optimal_ec) < 1e-8:
                    # 최적 EC(평균 생중량 최대) 강조
                    weight_colors.append("#FFD700")
                else:
                    weight_colors.append(base_color)

            fig_growth.add_trace(
                go.Bar(
                    x=x_labels,
                    y=ec_growth_summary_all["평균생중량"],
                    marker_color=weight_colors,
                    name="평균 생중량",
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

            fig_growth.update_xaxes(title_text="EC (dS/m)", row=1, col=1)
            fig_growth.update_xaxes(title_text="EC (dS/m)", row=1, col=2)
            fig_growth.update_xaxes(title_text="EC (dS/m)", row=1, col=3)

            fig_growth.update_yaxes(title_text="길이 (mm)", row=1, col=1)
            fig_growth.update_yaxes(title_text="길이 (mm)", row=1, col=2)
            fig_growth.update_yaxes(title_text="생중량 (g)", row=1, col=3)

            fig_growth.update_layout(
                height=450,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig_growth = apply_korean_font(fig_growth)
            st.plotly_chart(fig_growth, use_container_width=True)

            if optimal_ec is not None and optimal_school is not None:
                st.markdown(
                    "- 평균 생중량 기준으로 **최적 EC는 "
                    + format_float(optimal_ec, 1)
                    + " dS/m** 이며, 이는 **"
                    + str(optimal_school)
                    + "** 실험 조건에 해당합니다.\n"
                    "- 위 그래프에서 해당 EC의 막대는 **골드 색상**으로 강조되어 있습니다."
                )

        st.markdown("---")
        st.subheader("선택한 학교의 성장량 요약")

        if growth_selected is None or growth_selected.empty:
            st.info("선택한 학교에서 생육 데이터가 없습니다.")
        else:
            avg_shoot_sel = growth_selected[COL_SHOOT].mean()
            avg_root_sel = growth_selected[COL_ROOT].mean()
            avg_weight_sel = growth_selected[COL_FRESH].mean()
            count_sel = len(growth_selected)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("개체수", str(int(count_sel)))
            with col2:
                st.metric("평균 지상부 길이 (mm)", format_float(avg_shoot_sel, 1))
            with col3:
                st.metric("평균 지하부 길이 (mm)", format_float(avg_root_sel, 1))
            with col4:
                st.metric("평균 생중량 (g)", format_float(avg_weight_sel, 2))

            st.markdown(
                "아래 박스플롯은 선택한 학교 내 개체들의 분포를 보여줍니다."
            )

            fig_box_len = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("지상부 길이 분포", "지하부 길이 분포"),
                horizontal_spacing=0.12,
            )

            fig_box_len.add_trace(
                go.Box(
                    y=growth_selected[COL_SHOOT],
                    name="지상부",
                    boxmean="sd",
                ),
                row=1,
                col=1,
            )
            fig_box_len.add_trace(
                go.Box(
                    y=growth_selected[COL_ROOT],
                    name="지하부",
                    boxmean="sd",
                ),
                row=1,
                col=2,
            )

            fig_box_len.update_yaxes(title_text="길이 (mm)", row=1, col=1)
            fig_box_len.update_yaxes(title_text="길이 (mm)", row=1, col=2)
            fig_box_len.update_layout(
                height=400,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig_box_len = apply_korean_font(fig_box_len)
            st.plotly_chart(fig_box_len, use_container_width=True)

    # ====================== 탭 3 ====================== #
    with tab3:
        st.subheader("EC값에 따른 지상부·지하부 길이 비례 추정 (선형 회귀)")

        if ec_growth_summary_all.empty:
            st.error("EC별 성장 요약 데이터가 없어 관계를 분석할 수 없습니다.")
        else:
            df_ec = ec_growth_summary_all.sort_values("EC")
            x_ec = df_ec["EC"].astype(float).to_numpy()
            y_shoot = df_ec["평균지상부길이"].astype(float).to_numpy()
            y_root = df_ec["평균지하부길이"].astype(float).to_numpy()

            can_fit = len(df_ec) >= 2 and np.isfinite(x_ec).sum() >= 2

            if can_fit:
                coef_shoot = np.polyfit(x_ec, y_shoot, 1)
                coef_root = np.polyfit(x_ec, y_root, 1)
                x_line = np.linspace(float(x_ec.min()), float(x_ec.max()), 100)
                y_line_shoot = coef_shoot[0] * x_line + coef_shoot[1]
                y_line_root = coef_root[0] * x_line + coef_root[1]
            else:
                coef_shoot = None
                coef_root = None
                x_line = None
                y_line_shoot = None
                y_line_root = None

            colors_by_ec = []
            for ec_val in x_ec:
                sname = EC_TO_SCHOOL.get(float(ec_val))
                base_color = "#888888"
                if sname in SCHOOL_INFO:
                    base_color = SCHOOL_INFO[sname]["color"]
                colors_by_ec.append(base_color)

            fig_rel = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("EC vs 평균 지상부 길이", "EC vs 평균 지하부 길이"),
                horizontal_spacing=0.12,
            )

            # 지상부
            fig_rel.add_trace(
                go.Scatter(
                    x=x_ec,
                    y=y_shoot,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors_by_ec,
                        line=dict(width=1, color="black"),
                    ),
                    name="평균 지상부 길이",
                ),
                row=1,
                col=1,
            )
            if can_fit:
                fig_rel.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line_shoot,
                        mode="lines",
                        line=dict(color="#555555", dash="dash"),
                        name="지상부 선형 추정",
                    ),
                    row=1,
                    col=1,
                )

            # 지하부
            fig_rel.add_trace(
                go.Scatter(
                    x=x_ec,
                    y=y_root,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=colors_by_ec,
                        line=dict(width=1, color="black"),
                    ),
                    name="평균 지하부 길이",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            if can_fit:
                fig_rel.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line_root,
                        mode="lines",
                        line=dict(color="#555555", dash="dash"),
                        name="지하부 선형 추정",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )

            fig_rel.update_xaxes(title_text="EC (dS/m)", row=1, col=1)
            fig_rel.update_xaxes(title_text="EC (dS/m)", row=1, col=2)
            fig_rel.update_yaxes(title_text="평균 지상부 길이 (mm)", row=1, col=1)
            fig_rel.update_yaxes(title_text="평균 지하부 길이 (mm)", row=1, col=2)

            fig_rel.update_layout(
                height=450,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig_rel = apply_korean_font(fig_rel)
            st.plotly_chart(fig_rel, use_container_width=True)

            if can_fit:
                st.markdown(
                    "- 지상부 길이 선형 회귀식: **길이 ≈ "
                    + format_float(coef_shoot[0], 3)
                    + " × EC + "
                    + format_float(coef_shoot[1], 1)
                    + "**\n"
                    "- 지하부 길이 선형 회귀식: **길이 ≈ "
                    + format_float(coef_root[0], 3)
                    + " × EC + "
                    + format_float(coef_root[1], 1)
                    + "**"
                )
            else:
                st.info("유효한 회귀선을 계산하기에 데이터 포인트가 부족합니다.")

        st.markdown("---")
        st.subheader("지상부 길이 vs 지하부 길이 산점도 (선택한 학교 기준)")

        if growth_selected is None or growth_selected.empty:
            st.info("선택한 학교에서 생육 데이터가 없어 산점도를 표시할 수 없습니다.")
        else:
            fig_scatter = px.scatter(
                growth_selected,
                x=COL_SHOOT,
                y=COL_ROOT,
                color="EC",
                hover_data=["학교", COL_PLANT_ID],
                labels={
                    COL_SHOOT: "지상부 길이 (mm)",
                    COL_ROOT: "지하부 길이 (mm)",
                    "EC": "EC (dS/m)",
                },
                title="지상부 vs 지하부 길이 (개체 단위)",
            )
            fig_scatter.update_layout(
                height=450,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            fig_scatter = apply_korean_font(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)


if __name__ == "__main__":
    main()
