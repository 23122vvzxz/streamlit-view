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
#    col1, col2 = st.columns([3,1])

#    with col2:
#        image = Image.open('ui.png')
#        st.image(image, width=150)

    st.subheader("부유 진균 농도 예측")
    st.markdown(html_temp, unsafe_allow_html=True)

    menu = ['HOME', 'EDA', 'ML', 'About']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'HOME':
        st.subheader('HOME')
        st.markdown(dec_temp, unsafe_allow_html=True)
    elif choice == 'EDA':
        st.subheader('탐색적 자료 분석(EDA)')
        # EDA 관련 코드를 여기에 추가하세요
    elif choice == 'ML':
        st.subheader('머신러닝(ML)')
        # ML 관련 코드를 여기에 추가하세요
    else:
        st.subheader('About')
        # About 페이지 관련 코드를 여기에 추가하세요

if __name__ == "__main__":
    main()
