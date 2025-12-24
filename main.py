import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 한글 폰트 깨짐 방지
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# 데이터 로딩 함수
@st.cache_data
def load_data():
    data_path = Path("data")
    
    # CSV 파일 목록 찾기
    csv_files = [f for f in data_path.iterdir() if unicodedata.normalize("NFC", f.name) == f.name and f.suffix == '.csv']
    if len(csv_files) != 4:
        st.error("CSV 파일이 4개가 아닌 경우가 있습니다!")
        return None
    
    # 환경 데이터 로딩
    env_data = {}
    for csv_file in csv_files:
        school_name = csv_file.stem
        try:
            env_data[school_name] = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"파일 {csv_file.name}을 로딩하는데 문제가 발생했습니다: {e}")
            return None
    
    # 생육 결과 데이터 로딩
    xlsx_file = data_path / "4개교_생육결과데이터.xlsx"
    try:
        growth_data = pd.read_excel(xlsx_file, sheet_name=None)
    except Exception as e:
        st.error(f"생육 결과 데이터를 로딩하는데 문제가 발생했습니다: {e}")
        return None
    
    return env_data, growth_data

# 데이터 로딩
with st.spinner('데이터를 로딩 중입니다...'):
    data = load_data()
if data is None:
    st.stop()

env_data, growth_data = data

# 학교 선택 드롭다운
school_list = ['전체', '송도고', '하늘고', '아라고', '동산고']
school_choice = st.sidebar.selectbox("학교를 선택하세요", school_list)

# Tab 1: 실험 개요
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    st.header("연구 배경 및 목적")
    st.write("이 연구는 극지식물의 최적 EC 농도를 파악하고, 각 학교에서 측정된 환경 데이터 및 생육 결과를 비교 분석합니다.")
    
    # 학교별 EC 조건 표
    st.write("학교별 EC 조건")
    ec_table = {
        '학교명': ['송도고', '하늘고', '아라고', '동산고'],
        'EC 목표': [1.0, 2.0, 4.0, 8.0],
        '개체수': [len(growth_data['송도고']), len(growth_data['하늘고']), len(growth_data['아라고']), len(growth_data['동산고'])],
        '색상': ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    }
    ec_df = pd.DataFrame(ec_table)
    st.dataframe(ec_df)

    # 핵심 지표 카드
    total_plants = sum([len(growth_data[school]) for school in growth_data])
    avg_temp = np.mean([env_data[school]['temperature'].mean() for school in env_data])
    avg_humidity = np.mean([env_data[school]['humidity'].mean() for school in env_data])
    optimal_ec = 2.0  # 하늘고가 최적 EC

    st.metric("총 개체수", total_plants)
    st.metric("평균 온도", f"{avg_temp:.2f}°C")
    st.metric("평균 습도", f"{avg_humidity:.2f}%")
    st.metric("최적 EC 농도", f"{optimal_ec} EC (하늘고)")

with tab2:
    st.header("환경 데이터")
    
    # 학교별 환경 데이터 평균 비교
    fig = make_subplots(rows=2, cols=2, subplot_titles=["평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC"])
    
    avg_temp_values = [env_data[school]['temperature'].mean() for school in env_data]
    avg_humidity_values = [env_data[school]['humidity'].mean() for school in env_data]
    avg_ph_values = [env_data[school]['ph'].mean() for school in env_data]
    actual_ec_values = [env_data[school]['ec'].mean() for school in env_data]
    target_ec_values = [1.0, 2.0, 4.0, 8.0]
    
    fig.add_trace(go.Bar(x=list(env_data.keys()), y=avg_temp_values, name="평균 온도"), row=1, col=1)
    fig.add_trace(go.Bar(x=list(env_data.keys()), y=avg_humidity_values, name="평균 습도"), row=1, col=2)
    fig.add_trace(go.Bar(x=list(env_data.keys()), y=avg_ph_values, name="평균 pH"), row=2, col=1)
    fig.add_trace(go.Bar(x=list(env_data.keys()), y=actual_ec_values, name="실측 EC"), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(env_data.keys()), y=target_ec_values, mode="lines+markers", name="목표 EC", line=dict(color='black')), row=2, col=2)
    
    fig.update_layout(height=800, title_text="학교별 환경 데이터 평균 비교")
    st.plotly_chart(fig)

with tab3:
    st.header("생육 결과")
    
    # EC별 생육 비교
    fig_growth = make_subplots(rows=2, cols=2, subplot_titles=["평균 생중량", "평균 잎 수", "평균 지상부 길이", "개체수 비교"])
    
    avg_weight = [growth_data[school]['생중량(g)'].mean() for school in growth_data]
    avg_leaf_count = [growth_data[school]['잎 수(장)'].mean() for school in growth_data]
    avg_height = [growth_data[school]['지상부 길이(mm)'].mean() for school in growth_data]
    num_plants = [len(growth_data[school]) for school in growth_data]
    
    fig_growth.add_trace(go.Bar(x=list(growth_data.keys()), y=avg_weight, name="평균 생중량"), row=1, col=1)
    fig_growth.add_trace(go.Bar(x=list(growth_data.keys()), y=avg_leaf_count, name="평균 잎 수"), row=1, col=2)
    fig_growth.add_trace(go.Bar(x=list(growth_data.keys()), y=avg_height, name="평균 지상부 길이"), row=2, col=1)
    fig_growth.add_trace(go.Bar(x=list(growth_data.keys()), y=num_plants, name="개체수"), row=2, col=2)
    
    fig_growth.update_layout(height=800, title_text="EC별 생육 비교")
    st.plotly_chart(fig_growth)

    # EC별 평균 생중량 강조
    optimal_ec_school = '하늘고'
    optimal_weight = avg_weight[school_list.index(optimal_ec_school) - 1]
    st.subheader(f"최적 EC 농도({optimal_ec_school})에서의 평균 생중량: {optimal_weight:.2f} g")

# 학교별 생육 데이터 다운로드
with st.expander("학교별 생육 데이터 원본 + XLSX 다운로드"):
    buffer = io.BytesIO()
    growth_data["송도고"].to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    
    st.download_button(
        label="송도고 생육 데이터 다운로드",
        data=buffer,
        file_name="송도고_생육결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
