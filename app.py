import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import io

# GitHub에서 모델 로드
@st.cache_resource
def load_model():
    # GitHub의 원시 파일 URL을 입력하세요
    url = "https://github.com/23122vvzxz/streamlit-view/raw/main/logf_prediction_model.pkl"
    response = requests.get(url)
    if response.status_code == 200:
        model = pickle.loads(response.content)
        return model
    else:
        st.error("모델을 GitHub에서 로드하는 데 실패했습니다.")
        return None

def main():
    st.title('logf 예측 모델')
    
    # 모델 로드
    model = load_model()
    if model is None:
        st.stop()
    
    # 세션 상태 초기화
    if 'floor_indicator' not in st.session_state:
        st.session_state.floor_indicator = 0
    if 'tem' not in st.session_state:
        st.session_state.tem = 20.0
    if 'rh' not in st.session_state:
        st.session_state.rh = 50.0
    if 'd2_1' not in st.session_state:
        st.session_state.d2_1 = 0.0
    
    # 사용자 입력 받기
    floor_indicator = st.selectbox('Floor Indicator', [0, 1], index=int(st.session_state.floor_indicator), key='floor_indicator_select')
    tem = st.number_input('Temperature', value=float(st.session_state.tem), step=0.1, key='temperature_input')
    rh = st.number_input('Relative Humidity (%)', value=float(st.session_state.rh), min_value=0.0, max_value=100.0, step=0.1, key='humidity_input')
    d2_1 = st.number_input('d2', value=float(st.session_state.d2_1), step=0.01, key='d2_1_input')
    
    # 세션 상태 업데이트
    st.session_state.floor_indicator = int(floor_indicator)
    st.session_state.tem = float(tem)
    st.session_state.rh = float(rh)
    st.session_state.d2_1 = float(d2_1)
    
    # 예측 버튼
    if st.button('logf 예측하기', key='predict_button'):
        # 입력값으로 데이터프레임 생성
        input_data = pd.DataFrame([[floor_indicator, tem, rh, d2_1]], 
                                  columns=['floor_indicator', 'tem', 'rh', 'd2.1'])
        
        # floor_indicator를 one-hot encoding
        input_data = pd.get_dummies(input_data, columns=['floor_indicator'], prefix='floor_indicator')
        
        # 모델 학습 시 사용한 모든 one-hot encoded 컬럼이 있는지 확인
        expected_columns = ['floor_indicator_0', 'floor_indicator_1', 'tem', 'rh', 'd2.1']
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # 없는 컬럼은 0으로 채움
        
        # 컬럼 순서 조정
        input_data = input_data[expected_columns]
        
        # logf 예측 수행
        try:
            predicted_logf = model.predict(input_data)
            
            # 결과 출력
            st.write(f'예측된 logf 값: {predicted_logf[0]:.4f}')
        except Exception as e:
            st.error(f"예측 중 오류 발생: {str(e)}")
            st.write("입력 데이터:", input_data)
            st.write("모델 정보:", model)

if __name__ == '__main__':
    main()
