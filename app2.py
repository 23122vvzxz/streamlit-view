#utils.py
# -*- coding:UTF-8 -*-
p_lans = ['Python', 'Julia', 'Go', 'Rust']

# utils.py
# -*- coding:UTF-8 -*-

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# utils.py 내용을 직접 포함
html_temp = """
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax">과제명</th>
    <th class="tg-0lax">실내환경 중 부유 미생물 대사물질의 적정 관리 가이드라인 설정</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">과제 목표</td>
    <td class="tg-0lax">실내환경 중 부유 미생물 대사물질의 적정 관리기준안, 표준측정안 제시</td>
  </tr>
  <tr>
    <td class="tg-0lax">기관명</td>
    <td class="tg-0lax">경희대학교</td>
  </tr>
  <tr>
    <td class="tg-0lax">연구실</td>
    <td class="tg-0lax">환경안전성평가연구실</td>
  </tr>
</tbody>
</table>
"""

dec_temp = """
- 실내환경 실측 데이터를 활용하여 간단한 EDA 및 예측 모델을 구현한다.
"""

def main():
    col1, col2 = st.columns([3,1])

    with col2:
        image = Image.open('ui.png')
        st.image(image, width=150)



    menu = ['HOME', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'HOME':
        st.subheader('HOME')
        st.subheader("부유 진균 농도 예측")
        st.markdown(html_temp, unsafe_allow_html=True)
        st.markdown(dec_temp, unsafe_allow_html=True)
    elif choice == 'EDA':
        st.subheader('탐색적 자료 분석(EDA)')
        
        @st.cache_data
        def load_data():
            data = pd.read_csv('khu.csv')
            return data
        
        data = load_data()

        #EDA 옵션 선택
        eda_option = st.radio(
            "분석 항목:",
            ('데이터 미리보기', '기본 통계량', '결측치 확인', '히스토그램', '상관관계 히트맵', '산점도')
        )
      
        #데이터 미리보기
        if eda_option == '데이터 미리보기':
          st.write("데이터 미리보기:")
          st.write(data.head())

        elif eda_option == '기본 통계량':
        #기본 통계량
          st.write("기본 통계량:")
          st.write(data.describe())
        
        elif eda_option == '결측치 확인':
        #결측치 확인
          st.write("결측치 확인:")
          st.write(data.isnull().sum())

        elif eda_option == '히스토그램':
        #히스토그램
          st.write("히스토그램:")
          fig, axs = plt.subplots(figsize=(20,15), nrows=5, ncols=5)
          axs=axs.flatten()
          
          for i, column in enumerate(data.select_dtypes(include=['float64', 'int64']).columns):
            if i < len(axs):
              sns.histplot(data=data, x=column, kde=True, ax=axs[i])
              axs[i].set_title(column)
              axs[i].tick_params(axis='x', labelsize=8)
              axs[i].tick_params(axis='y', labelsize=8)

          for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

          plt.tight_layout()          
          st.pyplot(fig)

        elif eda_option == '상관관계 히트맵':
        #상관관계 분석
          st.write("상관관계 히트맵:")
          fig, ax = plt.subplots(figsize=(10,8))
          sns.heatmap(data.corr(), annot=False, cmap='coolwarm', ax=ax)
          st.pyplot(fig)

        elif eda_option == '산점도':
        #산점도
          st.write("산점도:")
          x_axis = st.selectbox('X축 선택', data.columns)
          y_axis = st.selectbox('Y축 선택', data.columns)
          color_option = st.selectbox('색상 구분', ['None'] + list(data.columns))
          fig, ax = plt.subplots(figsize =(10,6))
          if color_option == 'None':
              sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
          else:
              sns.scatterplot(data=data, x=x_axis, y=y_axis, hue=color_option, ax=ax)

          plt.xlabel(x_axis)
          plt.ylabel(y_axis)
          plt.title('plot')

          st.pyplot(fig)
      
    elif choice == 'ML':
        st.subheader('머신러닝(ML)')
        @st.cache_resource
        def load_scaler():
            return joblib.load('minmax_scaler.pkl')
        
        scaler = load_scaler()

        
        @st.cache_resource
        def load_model():
            with open('best_gbr_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        
        model = load_model()

        #데이터 로드
        @st.cache_data
        def load_data():
            data = pd.read_csv('khu.csv')
            return data
        
        data = load_data()

        #사용할 특성 선택 (예측 변수 제외)
        numerical_features = ['tem', 'rh', 'd2.1', 'co10.1']
        categorical_features = ['facility', 'floor_indicator']
        features = categorical_features + numerical_features

        #사용자 입력 받기
        st.write("예측을 위한 값을 입력하세요:")
        user_input = {}
      
        #범주형 변수
        user_input['facility'] = st.selectbox('시설 유형 [주차장(1), 지하철/고속철(2), 역사(3), 대형마트(4), 도서관(5), 어린이집(6), 리빙랩(7)]', options=data['facility'].unique())
        user_input['floor_indicator'] = st.selectbox('지하(0)/지상(1)', options=[0, 1])
      
        #수치형 변수
        user_input['tem'] = st.number_input('기온 (°C)', 
                                            min_value=float(data['tem'].min()),
                                            max_value=float(data['tem'].max()),
                                            value=float(data['tem'].mean()))
      
        user_input['rh'] = st.number_input('상대습도 (%)', 
                                           min_value=float(data['rh'].min()),
                                           max_value=float(data['rh'].max()),
                                           value=float(data['rh'].mean()))
      
        user_input['d2.1'] = st.number_input('PM10 농도 [log(μg/m³)]',
                                             min_value=float(data['d2.1'].min()),
                                             max_value=float(data['d2.1'].max()),
                                             value=float(data['d2.1'].mean()))
      
        user_input['co10.1'] = st.number_input('Coarse particle 농도 [log(μg/m³)]', 
                                               min_value=float(data['co10.1'].min()),
                                               max_value=float(data['co10.1'].max()),
                                               value=float(data['co10.1'].mean()))


        #예측 버튼
        if st.button('예측'):
            # 입력을 데이터프레임으로 변환
            input_df = pd.DataFrame([user_input])

            # 범주형 변수에 대해 원-핫 인코딩 적용
            input_encoded = pd.get_dummies(input_df, columns=categorical_features)

            #수치형 변수 스케일링
            input_scaled = scaler.transform(input_encoded[numerical_features])
            input_encoded[numerical_features] = input_scaled

            #모델의 특성 순서에 맞게 데이터 정령
            input_encoded = input_encoded[model.feature_names_in_]

            #예측
            prediction = model.predict(input_encoded)

            st.write(f'## 오늘의 부유 진균 농도 예측 결과: {prediction[0]} Log(Copy number/m³)')

#        st.write("특성 중요도:")
#        if hasattr(model, 'feature_importances_') and hasattr(model, 'feautre_names_in_'):
#            feature_importance = model.feature_importances_
#            feature_names = model.feature_names_in_
#            feature_importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
#            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            
#            fig, ax=plt.subplots()
#            sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
#            plt.title('Feature Importance')
#            st.pyplot(fig)
#        else:
#            st.write("이 모델은 특성 중요도를 제공하지 않습니다.")

    else:
        st.subheader('About')
        # About 페이지 관련 코드를 여기에 추가하세요

if __name__ == "__main__":
    main()
