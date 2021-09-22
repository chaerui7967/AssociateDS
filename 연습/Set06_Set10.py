# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:41:38 2021

@author: Administrator
"""

#%%

# =============================================================================
# =============================================================================
# # 문제 06 유형(DataSet_06.csv 이용)
#
# 구분자 : comma(“,”), 4,323 Rows, 19 Columns, UTF-8 인코딩

# 주택 관련 정보를 바탕으로 주택 가격을 예측해 보고자 한다. 
# 다음은 확보한 주택 관련 데이터로 총 19개 컬럼으로 구성되어
# 있다.

# 컬 럼 / 정 의 / Type
# id / 매물 번호 / Double
# date / 날짜 / String
# price / 거래 가격 / Double
# bedrooms / 방 개수 / Double
# bathrooms / 화장실 개수 (화장실은 있으나 샤워기 없는 경우 0.5로 처리) / Double
# sqft_living / 건축물 면적 / Double
# sqft_lot / 대지 면적 / Double
# floors / 건축물의 층수 / Double
# waterfront / 강변 조망 가능 여부 (0 / 1) / Double
# view / 경관 (나쁨에서 좋음으로 0 ~ 4로 표시) / Double
# condition / 관리 상태 (나쁨에서 좋음으로 1 ~ 5로 표시) / Double
# grade / 등급 (낮음에서 높음으로 1 ~ 13으로 표시) / Double
# sqft_above / 지상 면적 / Double
# sqft_basement / 지하실 면적 / Double
# yr_built / 건축 연도 / Double
# yr_renovated / 개축 연도 / Double
# zipcode / 우편번호 / Double
# sqft_living15 / 15개의 인근 주택의 평균 건물 면적 / Double
# sqft_lot15 / 15개의 인근 주택의 평균 대지 면적 / Double
# =============================================================================
# =============================================================================

import pandas as pd
import numpy as np

df6 = pd.read_csv('C:/Users/j/AssociateDS/Dataset/DataSet_06.csv')

#%%

# =============================================================================
# 1.강변 조망이 가능한지 여부(waterfront)에 따라 평균 주택 가격을 계산하고 조망이
# 가능한 경우와 그렇지 않은 경우의 평균 가격 차이의 절대값을 구하시오. 답은
# 소수점 이하는 버리고 정수부만 기술하시오. (답안 예시) 1234567
# =============================================================================

q11 = df6[df6.waterfront == 1]['price'].mean()
q12 = df6[df6.waterfront == 0]['price'].mean()

abs(q11 - q12)

# 답 1167272



#%%

# =============================================================================
# 2.price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, yr_built 등 7개의 변수 간의
# 상관분석을 수행하고 price와의 상관계수의 절대값이 가장 큰 변수와 가장 작은
# 변수를 차례로 기술하시오. (답안 예시) view, zipcode
# 
# =============================================================================

x_var=['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'yr_built']
q2 = df6[x_var].corr().drop('price')

q2['price'].abs().idxmax()
q2['price'].abs().idxmin()

# 답 'sqft_living', 'yr_built'



#%%

# =============================================================================
# 3. id, date, 그리고 zipcode를 제외한 모든 변수를 독립변수로, price를 종속변수로 하여
# 회귀분석을 수행하시오. 통계적 유의성을 갖지 못하는 독립변수를 제거하면 회귀
# 모형에 남는 변수는 모두
# 몇 개인가? 이 때 음의 회귀계수를 가지는 변수는 몇 개인가? (답안 예시) 5, 3
# =============================================================================

# =============================================================================
# (참고)
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from statsmodels.formula.api import ols
# =============================================================================

q3 = df6.drop(columns = ['id', 'date', 'zipcode'])

from statsmodels.formula.api import ols

x_var = df6.columns.drop(['id', 'date', 'zipcode', 'price'])

form = 'price~' + '+'.join(x_var)
ols1 = ols(form, q3).fit()
ols1.summary()

ols1.pvalues
ols1.params

(ols1.pvalues < 0.05).sum() - 1

(ols1.params[ols1.pvalues < 0.05] < 0).drop('Intercept').sum()
# 답 13 2


## Ols 사용
from statsmodels.api import OLS, add_constant

x = add_constant(q3[x_var])
ols2 = OLS(q3['price'],x).fit()
ols2.summary()

#%%

# =============================================================================
# =============================================================================
# # 문제 07 유형(DataSet_07.csv 이용)
#
# 구분자 : comma(“,”), 400 Rows, 9 Columns, UTF-8 인코딩
#
# 대학원 진학을 위하여 어떤 항목이 중요하게 영향을 미치는지
# 아래 데이터로 분석하고자 한다.

# 컬 럼 / 정 의 / Type
# Serial_No / 구분자 번호 / Double
# GRE / GRE 시험 성적 / Double
# TOEFL / TOEFL 시험 성적 / Double
# University_Rating / 대학 평가 그룹 (1 ~ 5) / Double
# SOP / 자기 소개서 점수 (1 ~ 5) / Double
# LOR / 추천서 점수 (1 ~ 5) / Double
# CGPA / 학부 평량 평점 (10점 만점 환산 점수) / Double
# Research / 연구 참여 경험 여부 (0 / 1) / Double
# Chance_of_Admit / 합격 가능성 / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# #2
# import scipy.stats as stats
# #3
# from sklearn.linear_model import LogisticRegression
# Solver = ‘liblinear’, random_state = 123
# =============================================================================

import pandas as pd
import numpy as np

df7 = pd.read_csv('C:/Users/j/AssociateDS/Dataset/DataSet_07.csv')

#%%

# =============================================================================
# 1. 합격 가능성에 GRE, TOEFL, CGPA 점수 가운데 가장 영향이 큰 것이 어떤 점수인지
# 알아 보기 위해서 상관 분석을 수행한다.
# - 피어슨(Pearson) 상관계수 값을 구한다.
# - Chance_of_Admit와의 가장 큰 상관계수 값을 가지는 항목의 상관계수를 소수점 넷째
# 자리에서 반올림하여 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================

x_var = ['GRE', 'TOEFL', 'CGPA', 'Chance_of_Admit']
q1 = df7[x_var].corr().drop('Chance_of_Admit')

q1['Chance_of_Admit'].nlargest(1)

# 답 0.873



#%%

# =============================================================================
# 2.GRE 점수의 평균 이상을 받은 그룹과 평균 미만을 받은 그룹의 CGPA 평균은 차이가
# 있는지
# 검정을 하고자 한다.
# - 적절한 검정 방법을 선택하고 양측 검정을 수행하시오 (등분산으로 가정)
# - 검정 결과, 검정통계량의 추정치를 소수점 셋째 자리에서 반올림하여 소수점 두 자리까지
# 기술하시오.
# (답안 예시) 1.23
# =============================================================================

m = df7['GRE'].mean()

q21 = df7[df7['GRE'] >= m]['CGPA']
q22 = df7[df7['GRE'] < m]['CGPA']

from scipy.stats import ttest_ind

q2 = ttest_ind(q21,q22, equal_var=True)

q2.statistic


# 답 19.44


#%%

# =============================================================================
# 3.Chance_of_Admit 확률이 0.5를 초과하면 합격으로, 이하이면 불합격으로 구분하고
# 로지스틱 회귀분석을 수행하시오.
# - 원데이터만 사용하고, 원데이터 가운데 Serial_No와 Label은 모형에서 제외
# - 각 설정값은 다음과 같이 지정하고, 언급되지 않은 사항은 기본 설정값을 사용하시오
# Seed : 123
# - 로지스틱 회귀분석 수행 결과에서 로지스틱 회귀계수의 절대값이 가장 큰 변수와 그 값을
# 기술하시오. 
# (로지스틱 회귀계수는 반올림하여 소수점 둘째 자리까지 / Intercept는 제외)
# (답안 예시) abc, 0.12
# =============================================================================

q3 = df7.copy()

q3['Chance_of_Admit'] = np.where(q3['Chance_of_Admit']>0.5,1,0)

x_var = q3.columns.drop(['Serial_No', 'Chance_of_Admit'])

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(random_state=12, fit_intercept=True,
                        solver='liblinear', C = 100000)
lg.fit(q3[x_var], q3['Chance_of_Admit'])

q3_out = pd.DataFrame({'x':x_var,
                       'coef':abs(lg.coef_).reshape(-1)})
q3_out['coef'].nlargest(1)
q3_out.sort_values('coef', ascending=False)

# 답 CGPA , 3.70

#%%

# =============================================================================
# =============================================================================
# # 문제 08 유형(DataSet_08.csv 이용)
#
# 구분자 : comma(“,”), 50 Rows, 5 Columns, UTF-8 인코딩
#
# 스타트업 기업들의 수익성에 대한 분석을 하기 위하여
# 아래와 같은 데이터를 입수하였다
#
# 
# 컬 럼 / 정 의 / Type
# RandD_Spend / 연구개발비 지출 / Double
# Administration / 운영관리비 지출 / Double
# Marketing_Spend / 마케팅비 지출 / Double
# State / 본사 위치 / String
# Profit / 이익 / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# import numpy as np
# #3
# from sklearn.linear_model import LinearRegression
# =============================================================================

import pandas as pd
import numpy as np

df8 = pd.read_csv('C:/Users/j/AssociateDS/Dataset/DataSet_08.csv')


#%%

# =============================================================================
# 1.각 주(State)별 데이터 구성비를 소수점 둘째 자리까지 구하고, 알파벳 순으로
# 기술하시오(주 이름 기준).
# (답안 예시) 0.12, 0.34, 0.54
# =============================================================================

df8.columns
# 'RandD_Spend', 'Administration',
# 'Marketing_Spend', 'State', 'Profit'

df8['State'].value_counts(normalize=True).sort_index().values

# 답 0.34, 0.32, 0.34

#%%

# =============================================================================
# 2.주별 이익의 평균을 구하고, 평균 이익이 가장 큰 주와 작은 주의 차이를 구하시오. 
# 차이값은 소수점 이하는 버리고 정수부분만 기술하시오. (답안 예시) 1234
# =============================================================================

tab = pd.pivot_table(df8, index = 'State', values= 'Profit')
tab.max() - tab.min()

# 답 14868

#%%

# =============================================================================
# 3.독립변수로 RandD_Spend, Administration, Marketing_Spend를 사용하여 Profit을 주별로
# 예측하는 회귀 모형을 만들고, 이 회귀모형을 사용하여 학습오차를 산출하시오.
# - 주별로 계산된 학습오차 중 MAPE 기준으로 가장 낮은 오차를 보이는 주는 어느
# 주이고 그 값은 무엇인가? (반올림하여 소수점 둘째 자리까지 기술하시오)
# - (MAPE = Σ ( | y - y ̂ | / y ) * 100/n )
# (답안 예시) ABC, 1.56
# =============================================================================

state = df8.State.unique()
x_var = ['RandD_Spend', 'Administration', 'Marketing_Spend']
out = []

from sklearn.linear_model import LinearRegression

for i in state:
       temp = df8[df8.State == i]
       lm = LinearRegression().fit(temp[x_var], temp['Profit'])
       pred = lm.predict(temp[x_var])
       mape = (abs(temp['Profit'] - pred)/temp['Profit']).sum() * 100 / len(temp)
       out += [[i, mape]]

out = pd.DataFrame(out, columns=['x','mape'])

out.sort_values('mape')

# 답 Florida   5.71

#%%

# =============================================================================
# =============================================================================
# # 문제 09 유형(DataSet_09.csv 이용)
#
# 구분자 : comma(“,”), 2000 Rows, 16 Columns, UTF-8 인코딩
#
# 항공사에서 고객만족도 조사를 하고 서비스 개선에 활용하고자
# 아래와 같은 데이터를 준비하였다.
#
# 컬 럼 / 정 의 / Type
# satisfaction / 서비스 만족 여부 / String
# Gender / 성별 / String
# Age / 나이 / Double
# Customer_Type / 고객 타입 / String
# Class / 탑승 좌석 등급 / String
# Flight_Distance / 비행 거리 / Double
# Seat_comfort / 좌석 안락도 점수 / Double
# Food_and_Drink / 식사와 음료 점수 / Double
# Inflight_wifi_service / 기내 와이파이 서비스 점수 / Double
# Inflight_entertainment / 기내 엔터테인먼트 서비스 점수 / Double
# Onboard_service / 탑승 서비스 점수 / Double
# Leg_room_service / 다리 공간 점수 / Double
# Baggage_handling / 수하물 취급 점수 / Double
# Cleanliness / 청결도 점수 / Double
# Departure_Daly_in_Minutes / 출발 지연 (분) / Double
# Arrival_Delay_in_Minutes / 도착 지연 (분) / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# import scipy.stats as stats
# #3
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# 
# =============================================================================

import pandas as pd
import numpy as np

df9 = pd.read_csv('C:/Users/j/AssociateDS/Dataset/DataSet_09.csv')


#%%

# =============================================================================
# 1.데이터 타입을 위 표에 정의된 타입으로 전처리를 한 후, 데이터 파일 내에 결측값은
# 총 몇 개인가? (답안 예시) 1
# =============================================================================


df9.info()
df9.isna().sum().sum()

# 답 5

#%%
# =============================================================================
# 2.다음에 제시된 데이터 처리를 하고 카이제곱 독립성 검정을 수행하시오.
# - 결측값이 있다면 해당 행을 제거하시오.
# - 나이는 20 이하이면 10, 30 이하이면 20, 40 이하이면 30, 50 이하이면 40, 60 이하이면 50, 
# 60 초과는 60으로 변환하여 Age_gr으로 파생변수를 생성하시오.
# - Age_gr, Gender, Customer_Type, Class 변수가 satisfaction에 영향이 있는지 카이제곱
# 독립성 검정을 수행하시오. 
# - 연관성이 있는 것으로 파악된 변수의 검정통계량 추정치를 정수 부분만 기술하시오. 
# (답안 예시) 123
# =============================================================================

from scipy.stats import chi2_contingency

q2 = df9.dropna()

q2['Age_gr']=np.where(q2.Age <= 20, 10,
                np.where(q2.Age <= 30, 20,
                   np.where(q2.Age <= 40, 30,
                     np.where(q2.Age <= 50, 40,
                        np.where(q2.Age <= 60, 50,  60)))))

var = ['Age_gr', 'Gender', 'Customer_Type', 'Class']

out = []

for i in var:
       tab = pd.crosstab(index=q2[i], columns=q2.satisfaction)
       chi, p, *_ = chi2_contingency(tab)
       out.append([i,chi,p])

out = pd.DataFrame(out, columns=['x', 'chi', 'p'])

out[out.p < 0.05]

# 답 1066

#%%

# =============================================================================
# 3.고객 만족도를 라벨로 하여 다음과 같이 로지스틱 회귀분석을 수행하시오. 
# - 결측치가 포함된 행은 제거
# - 데이터를 7대 3으로 분리 (Seed = 123)
# - 아래의 11개 변수를 Feature로 사용
# Flight_Distance, Seat_comfort, Food_and_drink, Inflight_wifi_service, 
# Inflight_entertainment,Onboard_service, Leg_room_service, Baggage_handling,
# Cleanliness, Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes
# 
# - Seed = 123, 이외의 항목은 모두 Default 사용
# - 예측 정확도를 측정하고 dissatisfied의 f1 score를 소수점 넷째 자리에서 반올림하여
# 소수점 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

q3 = df9.dropna()
x_var=['Flight_Distance', 'Seat_comfort', 'Food_and_drink',
       'Inflight_wifi_service','Inflight_entertainment', 'Onboard_service',
       'Leg_room_service', 'Baggage_handling', 'Cleanliness',
       'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']

x_train, x_test, y_train, y_test = train_test_split(q3[x_var], q3['satisfaction'],
                                                    test_size=0.3, random_state= 123)
lg = LogisticRegression(solver='liblinear', random_state= 123).fit(x_train, y_train)
pred = lg.predict(x_test)

f1_score(y_test, pred, pos_label='dissatisfied')
print(classification_report(y_test, pred))
FF, FM, MF, MM = confusion_matrix(y_test, pred).ravel()

# 답 0.778


#%%

# =============================================================================
# =============================================================================
# # 문제 10 유형(DataSet_10.csv 이용)
#
# 구분자 : comma(“,”), 1538 Rows, 6 Columns, UTF-8 인코딩

# 중고 자동차 가격에 대한 분석을 위하여 아래와 같은 데이터를
# 확보하였다.

# 컬 럼 / 정 의 / Type
# model / 모델명 / String
# engine_power / 엔진 파워 / Double
# age_in_days / 운행 일수 / Double
# km / 운행 거리 / Double
# previous_owners / 이전 소유자 수 / Double
# price / 중고차 가격 / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# import scipy.stats as ststs
# #3
# from sklearn.linear_model import LinearRegression
# =============================================================================

import pandas as pd
import numpy as np

df10 = pd.read_csv('C:/Users/j/AssociateDS/Dataset/DataSet_10.csv')
df10 = df10.dropna(axis=1, how='all')

#%%

# =============================================================================
# 1.이전 소유자 수가 한 명이고 엔진 파워가 51인 차에 대해 모델별 하루 평균 운행
# 거리를 산출하였을 때 가장 낮은 값을 가진 모델이 가장 큰 값을 가진 모델에 대한
# 비율은 얼마인가? 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지
# 기술하시오.
# (모델별 평균 → 일평균 → 최대최소 비율 계산) (답안 예시) 0.12
# =============================================================================

q1 = df10.copy()
q1 = q1[(q1.previous_owners == 1) & (q1.engine_power == 51)]

tab = pd.pivot_table(q1, index='model', values=['km','age_in_days'])
tab['km_per'] = tab['km']/ tab['age_in_days']

tab['km_per'].min()/ tab['km_per'].max()

# 답 0.97


#%%

# =============================================================================
# 2.운행 일수에 대한 운행 거리를 산출하고, 위 1번 문제에서 가장 큰 값을 가지고 있던
# 모델과 가장 낮은 값을 가지고 있던 모델 간의 운행 일수 대비 운행거리 평균이 다른지
# 적절한 통계 검정을 수행하고 p-value를 소수점 세자리 이하는 버리고 소수점
# 두자리까지 기술하고 기각 여부를 Y / N로 답하시오. (등분산을 가정하고 equal_var = 
# True / var.equal = T로 분석을 실행하시오.)
# (답안 예시) 0.23, Y
# =============================================================================

ma = tab['km_per'].idxmax()
mi = tab['km_per'].idxmin()

q2 = df10.copy()
q2['km_per'] = q2['km'] / q2['age_in_days']

max_q2 = q2[q2.model == ma]['km_per']
min_q2 = q2[q2.model == mi]['km_per']

from scipy.stats import ttest_ind
ttest_ind(max_q2, min_q2, equal_var=True)

# 답 0.13 , N

#%%

# =============================================================================
# 3.독립변수로 engine_power, age_in_days, km를 사용하고 종속변수로 price를 사용하여
# 모델별 선형회귀분석을 수행하고, 산출된 모형을 사용하여 다음과 같은 조건의
# 중고차에 대한 가격을 예측하고 예측된 가격을 정수부만 기술하시오.
# - model : pop / engine_power : 51 / age_in_days : 400 / km : 9500 / previous_owners : 2

# (답안 예시) 12345
# =============================================================================

q3 = df10.copy()
x_var= ['engine_power', 'age_in_days', 'km']
model = q3.model.unique()

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(q3[x_var], q3['price'])
new_data = np.array([[51,400,9500]])
pred = lr.predict(new_data)
pred

# 답 10469

