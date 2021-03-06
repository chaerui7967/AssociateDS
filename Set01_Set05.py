# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 12:53:16 2021

@author: Administrator
"""
## 함수 정보 보기 ctrl+i
#%%

# =============================================================================
# =============================================================================
# # 문제 01 유형(DataSet_01.csv 이용)
#
# 구분자 : comma(“,”), 4,572 Rows, 5 Columns, UTF-8 인코딩
# 
# 글로벌 전자제품 제조회사에서 효과적인 마케팅 방법을 찾기
# 위해서 채널별 마케팅 예산과 매출금액과의 관계를 분석하고자
# 한다.
# 컬 럼 / 정 의  /   Type
# TV   /     TV 마케팅 예산 (억원)  /   Double
# Radio / 라디오 마케팅 예산 (억원)  /   Double
# Social_Media / 소셜미디어 마케팅 예산 (억원)  / Double
# Influencer / 인플루언서 마케팅
# (인플루언서의 영향력 크기에 따라 Mega / Macro / Micro / 
# Nano) / String

# SALES / 매출액 / Double
# =============================================================================
# =============================================================================
# pandas, scipy, numpy, sklearn, statsmodels

import pandas as pd
import numpy as np
df = pd.read_csv('./Dataset/DataSet_01.csv', encoding='UTF-8')


#%%

# =============================================================================
# 1. 데이터 세트 내에 총 결측값의 개수는 몇 개인가? (답안 예시) 23
# =============================================================================
# 답
df.isna().sum().sum()  # 열별 결측치 수 확인
# or
df.isnull().sum().sum()

# 비교
df.isna().sum(axis=1)  # 행별 결측치 수
df.isna().any(axis=1).sum()




#%%

# =============================================================================
# 2. TV, Radio, Social Media 등 세 가지 다른 마케팅 채널의 예산과 매출액과의 상관분석을
# 통하여 각 채널이 매출에 어느 정도 연관이 있는지 알아보고자 한다. 
# - 매출액과 가장 강한 상관관계를 가지고 있는 채널의 상관계수를 소수점 5번째
# 자리에서 반올림하여 소수점 넷째 자리까지 기술하시오. (답안 예시) 0.1234
# =============================================================================

# 데이터의 종류 수치형
# 답
print(np.round(df[['TV', 'Radio', 'Social_Media', 'Sales']].corr().abs()['Sales'].sort_values(ascending = False)[1],4))
# or
df.corr()
# or
print(np.round(df[['TV', 'Radio', 'Social_Media', 'Sales']].corr().abs()['Sales'].nlargest(2)[1],4))




#%%

# =============================================================================
# 3. 매출액을 종속변수, TV, Radio, Social Media의 예산을 독립변수로 하여 회귀분석을
# 수행하였을 때, 세 개의 독립변수의 회귀계수를 큰 것에서부터 작은 것 순으로
# 기술하시오. 
# - 분석 시 결측치가 포함된 행은 제거한 후 진행하며, 회귀계수는 소수점 넷째 자리
# 이하는 버리고 소수점 셋째 자리까지 기술하시오. (답안 예시) 0.123
# from sklearn.linear_model import LinearRegression 사용하시오.
# =============================================================================

from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant  # 선형, 절편추가
from statsmodels.formula.api import ols  # R기반 회귀분석

# 결측치 제거
df1 = df.dropna()

# sklearn 사용
lm = LinearRegression(fit_intercept=True)  # 절편 유무
lm.fit(df1.drop(columns = ['Influencer', 'Sales']), df1.Sales)  # 절편 유무

dir(lm) # 변수 이름 확인 coef_ : 회귀계수 확인, intercept_ : 절편
# 답
np.round(lm.coef_,3)

# statsmodels 사용
x=df1.drop(columns=['Influencer', 'Sales'])
xx=add_constant(x)  # 상수항 추가 
y=df1['Sales']

ols1 = OLS(y, xx).fit() 
ols1.summary()

var_list=list(df1.columns[:-1])
form = 'Sales~'+'+'.join(var_list)

ols2 = ols(form, data = df1).fit()
ols2.summary()

#%%

# =============================================================================
# =============================================================================
# # 문제 02 유형(DataSet_02.csv 이용)
# 구분자 : comma(“,”), 200 Rows, 6 Columns, UTF-8 인코딩

# 환자의 상태와 그에 따라 처방된 약에 대한 정보를 분석하고자한다
# 
# 컬 럼 / 정 의  / Type
# Age  / 연령 / Integer
# Sex / 성별 / String
# BP / 혈압 레벨 / String
# Cholesterol / 콜레스테롤 레벨 /  String
# Na_to_k / 혈액 내 칼륨에 대비한 나트륨 비율 / Double
# Drug / Drug Type / String
# =============================================================================
# =============================================================================

dataset2 = pd.read_csv('./Dataset/DataSet_02.csv', encoding='UTF-8')
dataset2.columns
dataset2.dtypes
dataset2.shape


#%%

# =============================================================================
# 1.해당 데이터에 대한 EDA를 수행하고, 여성으로 혈압이 High, Cholesterol이 Normal인
# 환자의 전체에 대비한 비율이 얼마인지 소수점 네 번째 자리에서 반올림하여 소수점 셋째
# 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================

# 빈도 --> 크로스탭
# 답
Q1=pd.crosstab(index=[dataset2.Sex, dataset2.BP],
               columns=dataset2.Cholesterol,
               normalize=True)

# pivot
?pd.pivot_table
Q1_p = pd.pivot_table(index=['Sex', 'BP'],
                      columns=['Cholesterol'],
                      values = ['Drug'],
                      data = dataset2,
                      aggfunc = 'count')  # normalize 키워드 없음

Q1_p/len(dataset2) # 답

Q2_p = pd.pivot_table(index=['Sex', 'BP'],
                      columns=['Cholesterol'],
                      values = ['Age', 'Drug'],
                      data = dataset2,
                      aggfunc = ['mean','count']
                      ## {'Age':np.mean, 'Drug': 'count'}
                      )

#%%

# =============================================================================
# 2. Age, Sex, BP, Cholesterol 및 Na_to_k 값이 Drug 타입에 영향을 미치는지 확인하기
# 위하여 아래와 같이 데이터를 변환하고 분석을 수행하시오. 
# - Age_gr 컬럼을 만들고, Age가 20 미만은 ‘10’, 20부터 30 미만은 ‘20’, 30부터 40 미만은
# ‘30’, 40부터 50 미만은 ‘40’, 50부터 60 미만은 ‘50’, 60이상은 ‘60’으로 변환하시오. 
# - Na_K_gr 컬럼을 만들고 Na_to_k 값이 10이하는 ‘Lv1’, 20이하는 ‘Lv2’, 30이하는 ‘Lv3’, 30 
# 초과는 ‘Lv4’로 변환하시오.
# - Sex, BP, Cholesterol, Age_gr, Na_K_gr이 Drug 변수와 영향이 있는지 독립성 검정을
# 수행하시오.
# - 검정 수행 결과, Drug 타입과 연관성이 있는 변수는 몇 개인가? 연관성이 있는 변수
# 가운데 가장 큰 p-value를 찾아 소수점 여섯 번째 자리 이하는 버리고 소수점 다섯
# 번째 자리까지 기술하시오.
# (답안 예시) 3, 1.23456
# =============================================================================

# Tip. 변수 생성
# 변수변환
Q2 = dataset2.copy()

Q2['Age_gr']=np.where(Q2.Age<20,'10',
                      np.where(Q2.Age<30,'20',
                               np.where(Q2.Age<40,'30',
                                        np.where(Q2.Age<50,'40',
                                                 np.where(Q2.Age<60,'50','60')))))
# cut 사용
age_gr=pd.cut(Q2.Age, [0,20,30,40,50,60, Q2.Age.max()+1],
              right=False,
              labels=['10','20','30','40','50','60'])
# < x <= 이므로 최소값 포함하려면 min-1 or include_lowest=True 설정

# Na_K_gr
Q2['Na_K_gr'] = np.where(Q2.Na_to_K <= 10, 'Lv1',
                         np.where(Q2.Na_to_K <= 20, 'Lv2',
                                  np.where(Q2.Na_to_K <= 30, 'Lv3','Lv4')))

import scipy.stats as sc

# 카이제곱 검정은 입력값이 빈도 테이블로 들어가도록 변경후 적용
Q2.columns
# ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug', 'Age_gr', 'Na_K_gr']

Age_t = pd.crosstab(Q2.Age_gr, Q2.Drug)
out_Age_t = sc.chi2_contingency(Age_t)
print('Age_gr : ', out_Age_t[1])

Q2_temp = []

for i in ['Age_gr','Sex', 'BP', 'Cholesterol', 'Na_K_gr']:
    temp = pd.crosstab(Q2[i], Q2.Drug)
    out_temp = sc.chi2_contingency(temp)
    # print(i, ' : ', out_temp[1])
    # Q2_temp.append(out_temp[1])
    Q2_temp = Q2_temp + [[i, out_temp[1]]]

Q2_temp=pd.DataFrame(Q2_temp)

print(Q2_temp)

Q2_temp.columns = ['var', 'pvalues']
Q2_temp2 = Q2_temp[Q2_temp.pvalues < 0.05]

print(Q2_temp2)

Q2_temp2.sort_values(by='pvalues').tail(1)

# 답
# var   pvalues
# 0  Age_gr  0.000701

#%%

# =============================================================================
# 3.Sex, BP, Cholesterol 등 세 개의 변수를 다음과 같이 변환하고 의사결정나무를 이용한
# 분석을 수행하시오.
# - Sex는 M을 0, F를 1로 변환하여 Sex_cd 변수 생성
# - BP는 LOW는 0, NORMAL은 1 그리고 HIGH는 2로 변환하여 BP_cd 변수 생성
# - Cholesterol은 NORMAL은 0, HIGH는 1로 변환하여 Ch_cd 생성
# - Age, Na_to_k, Sex_cd, BP_cd, Ch_cd를 Feature로, Drug을 Label로 하여 의사결정나무를
# 수행하고 Root Node의 split feature와 split value를 기술하시오. 
# 이 때 split value는 소수점 셋째 자리까지 반올림하여 기술하시오. (답안 예시) Age, 
# 12.345
# =============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

Q3=dataset2.copy()

# 변수 변환
Q3['Sex_cd']=np.where(Q3.Sex == 'M', 0, 1)
Q3['BP_cd']=np.where(Q3.BP == 'LOW', 0,
                     np.where(Q3.BP=='NORMAL',1,2))
Q3['Ch_cd']=np.where(Q3.Cholesterol == 'NORMAL', 0, 1)

x_var=['Age', 'Na_to_K', "Sex_cd", 'BP_cd', 'Ch_cd']

dt=DecisionTreeClassifier().fit(Q3[x_var], Q3.Drug)

plot_tree(dt, feature_names = x_var, class_names=Q3.Drug.unique())

export_text(dt, feature_names = x_var, decimals=3)

# 답
# Na_to_K <= 14.829


#%%

# =============================================================================
# =============================================================================
# # 문제 03 유형(DataSet_03.csv 이용)
# 
# 구분자 : comma(“,”), 5,001 Rows, 8 Columns, UTF-8 인코딩
# 안경 체인을 운영하고 있는 한 회사에서 고객 사진을 바탕으로 안경의 사이즈를
# 맞춤 제작하는 비즈니스를 기획하고 있다. 우선 데이터만으로 고객의 성별을
# 파악하는 것이 가능할 지를 연구하고자 한다.
#
# 컬 럼 / 정 의 / Type
# long_hair / 머리카락 길이 (0 – 길지 않은 경우 / 1 – 긴
# 경우) / Integer
# forehead_width_cm / 이마의 폭 (cm) / Double
# forehead_height_cm / 이마의 높이 (cm) / Double
# nose_wide / 코의 넓이 (0 – 넓지 않은 경우 / 1 – 넓은 경우) / Integer
# nose_long / 코의 길이 (0 – 길지 않은 경우 / 1 – 긴 경우) / Integer
# lips_thin / 입술이 얇은지 여부 0 – 얇지 않은 경우 / 1 –
# 얇은 경우) / Integer
# distance_nose_to_lip_long / 인중의 길이(0 – 인중이 짧은 경우 / 1 – 인중이
# 긴 경우) / Integer
# gender / 성별 (Female / Male) / String
# =============================================================================
# =============================================================================

dataset3 = pd.read_csv('./Dataset/DataSet_03.csv')
dataset3.columns
dataset3.shape
dataset3.dtypes

#%%

# =============================================================================
# 1.이마의 폭(forehead_width_cm)과 높이(forehead_height_cm) 사이의
# 비율(forehead_ratio)에 대해서 평균으로부터 3 표준편차 밖의 경우를 이상치로
# 정의할 때, 이상치에 해당하는 데이터는 몇 개인가? (답안 예시) 10
# =============================================================================

# 변수 생성(forehead_ration)
Q1 = dataset3.copy()
Q1['forehead_ratio'] = Q1['forehead_width_cm'] / Q1['forehead_height_cm']
m = Q1['forehead_ratio'].mean()
sd = Q1['forehead_ratio'].std()

cnt = len(Q1[Q1['forehead_ratio'] > m+3*sd]) + len(Q1[Q1['forehead_ratio'] < m-3*sd])
cnt

## 풀이
## 동일
LL=m - (3*sd)
UU=m + (3*sd)
Q1_out = Q1[(Q1['forehead_ratio'] < LL) | (Q1['forehead_ratio'] > UU)]
Q1_ans = len(Q1_out)
print(Q1_ans)

# 답
# 3


#%%

# =============================================================================
# 2.성별에 따라 forehead_ratio 평균에 차이가 있는지 적절한 통계 검정을 수행하시오.
# - 검정은 이분산을 가정하고 수행한다. 
# - 검정통계량의 추정치는 절대값을 취한 후 소수점 셋째 자리까지 반올림하여
# 기술하시오.
# - 신뢰수준 99%에서 양측 검정을 수행하고 결과는 귀무가설 기각의 경우 Y로, 그렇지
# 않을 경우 N으로 답하시오. (답안 예시) 1.234, Y
# =============================================================================


# 집단 수 확인
# 2개 ==>  t-test --> 독립표본 vs 대응표본
#       --> 대응의 경우 분산를 가정하지 않음
# else ==> anova

# 독립인 두 집단간의 평균차이

from scipy.stats import ttest_1samp, ttest_ind, ttest_rel 
# 독립 1표본, 독립 2표본, 대응표본

ttest_ind(Q1.forehead_ratio[Q1.gender=='Male'],
          Q1.forehead_ratio[Q1.gender=='Female'],
          equal_var=False)  # 이분산 가정 
            # conf_level=0.99

# 답
# Ttest_indResult(statistic=2.9994984197511543
#                pvalue=0.0027186702390657176)

# 2.999 Y


#%%

# =============================================================================
# 3.주어진 데이터를 사용하여 성별을 구분할 수 있는지 로지스틱 회귀분석을 적용하여
# 알아 보고자 한다. 
# - 데이터를 7대 3으로 나누어 각각 Train과 Test set로 사용한다. 이 때 seed는 123으로
# 한다.
# - 원 데이터에 있는 7개의 변수만 Feature로 사용하고 gender를 label로 사용한다.
# (forehead_ratio는 사용하지 않음)
# - 로지스틱 회귀분석 예측 함수와 Test dataset를 사용하여 예측을 수행하고 정확도를
# 평가한다. 이 때 임계값은 0.5를 사용한다. 
# - Male의 Precision 값을 소수점 둘째 자리까지 반올림하여 기술하시오. (답안 예시) 
# 0.12
# 
# 
# (참고) 
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# train_test_split 의 random_state = 123
# =============================================================================


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score

# 데이터 분리
train, test = train_test_split(dataset3, test_size=0.3,
                                                    random_state=123)
model1 = LogisticRegression().fit(train.drop(columns='gender'),
                                  train.gender)

pred = model1.predict(test.drop(columns='gender'))

pred_p = model1.predict_proba(test.drop(columns='gender'))
# 기본 임계값은 0.5 바꾸고 싶다면 /임계값해서 몫이 있으면 1 없으면 0

print(classification_report(test.gender, pred))

precision_score(test.gender, pred, pos_label='Male')

# 답
# 0.9596354166666666

# 임계값 변경
(pred_p[:,1] > 0.7).sum()


#%%

# =============================================================================
# =============================================================================
# # 문제 04 유형(DataSet_04.csv 이용)
#
#구분자 : comma(“,”), 6,718 Rows, 4 Columns, UTF-8 인코딩

# 한국인의 식생활 변화가 건강에 미치는 영향을 분석하기에 앞서 육류
# 소비량에 대한 분석을 하려고 한다. 확보한 데이터는 세계 각국의 1인당
# 육류 소비량 데이터로 아래와 같은 내용을 담고 있다.

# 컬 럼 / 정 의 / Type
# LOCATION / 국가명 / String
# SUBJECT / 육류 종류 (BEEF / PIG / POULTRY / SHEEP) / String
# TIME / 연도 (1990 ~ 2026) / Integer
# Value / 1인당 육류 소비량 (KG) / Double
# =============================================================================
# =============================================================================

# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# from scipy.stats import ttest_rel
# #3
# from sklearn.linear_model import LinearRegression

dataset4 = pd.read_csv('./Dataset/DataSet_04.csv')

#%%

# =============================================================================
# 1.한국인의 1인당 육류 소비량이 해가 갈수록 증가하는 것으로 보여 상관분석을 통하여
# 확인하려고 한다. 
# - 데이터 파일로부터 한국 데이터만 추출한다. 한국은 KOR로 표기되어 있다.
# - 년도별 육류 소비량 합계를 구하여 TIME과 Value간의 상관분석을 수행하고
# 상관계수를 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지만 기술하시오. 
# (답안 예시) 0.55
# =============================================================================

dataset4.columns
# ['LOCATION', 'SUBJECT', 'TIME', 'Value']
Q1 = dataset4.copy()

Q1 = Q1[Q1.LOCATION == 'KOR']
Q1_out = pd.pivot_table(data=Q1, index='TIME',
                        values='Value',
                        aggfunc='sum')
Q1_out.reset_index(inplace=True)

Q1_out.corr()['Value']['TIME']

# 답
# 0.96


#%%

# =============================================================================
# 2. 한국 인근 국가 가운데 식생의 유사성이 상대적으로 높은 일본(JPN)과 비교하여, 연도별
# 소비량에 평균 차이가 있는지 분석하고자 한다.
# - 두 국가의 육류별 소비량을 연도기준으로 비교하는 대응표본 t 검정을 수행하시오.
# - 두 국가 간의 연도별 소비량 차이가 없는 것으로 판단할 수 있는 육류 종류를 모두
# 적으시오. (알파벳 순서) (답안 예시) BEEF, PIG, POULTRY, SHEEP
# =============================================================================

from scipy.stats import ttest_rel

Q2 = dataset4[dataset4.LOCATION.isin(['KOR', 'JPN'])]
Q2_out = pd.pivot_table(data=Q2, index=['TIME', 'SUBJECT'],
                        columns = 'LOCATION',
                        values = 'Value')
Q2_out=Q2_out.dropna()
Q2_out = Q2_out.reset_index()

sub_list = Q2_out.SUBJECT.unique()
# ['POULTRY', 'SHEEP', 'BEEF', 'PIG']

Q2_out2 =[]

for i in sub_list:
    temp = Q2_out[Q2_out.SUBJECT==i]
    pvalue = ttest_rel(temp['KOR'], temp['JPN']).pvalue
    Q2_out2 = Q2_out2 + [[i, pvalue]]
    
Q2_out2 = pd.DataFrame(Q2_out2, columns = ['sub', 'pvalue'])

Q2_out2[Q2_out2.pvalue >= 0.05]

# 답
# POULTRY



#%%

# =============================================================================
# 3.(한국만 포함한 데이터에서) Time을 독립변수로, Value를 종속변수로 하여 육류
# 종류(SUBJECT) 별로 회귀분석을 수행하였을 때, 가장 높은 결정계수를 가진 모델의
# 학습오차 중 MAPE를 반올림하여 소수점 둘째 자리까지 기술하시오. (답안 예시) 21.12
# (MAPE : Mean Absolute Percentage Error, 평균 절대 백분율 오차)
# (MAPE = Σ ( | y - y ̂ | / y ) * 100/n ))
# 
# =============================================================================


from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS, add_constant
from sklearn.metrics import r2_score, mean_absolute_error

Q3 = Q2_out.drop(columns = 'JPN')

Q3_out=[]

sub_list = Q3.SUBJECT.unique()

for i in sub_list:
    temp = Q3[Q3.SUBJECT == i]  # 입력변수는 2차원구조로 입력해야함
    lm = LinearRegression().fit(temp[['TIME']], temp.KOR)  # Q3[''] 는 1차원,
    # Q3[['']] 2차원
    r2_Score = lm.score(temp[['TIME']], temp.KOR)
    Q3_out = Q3_out + [[i,r2_Score]]

Q3_out = pd.DataFrame(Q3_out, columns = ['sub', 'r2_score'])
# 'POULTRY'

temp = Q3[Q3.SUBJECT == 'POULTRY']
lm2 = LinearRegression().fit(temp[['TIME']], temp.KOR)
pred = lm2.predict(temp[['TIME']])

# MAPE = Σ ( | y - y ̂ | / y ) * 100/n 

mape = sum(abs(temp.KOR - pred) / temp.KOR) * 100 / len(temp)
# 답
# 5.783357902874552

#%%
# grobal 사용

Q3_out=[]

sub_list = Q3.SUBJECT.unique()

for i in sub_list:
    temp = Q3[Q3.SUBJECT == i]  # 입력변수는 2차원구조로 입력해야함
    globals()['lm_'+str(i)] = LinearRegression().fit(temp[['TIME']], temp.KOR)  # Q3[''] 는 1차원,
    # Q3[['']] 2차원
    r2_Score = eval('lm_'+str(i)).score(temp[['TIME']], temp.KOR)
    Q3_out = Q3_out + [[i,r2_Score]]

Q3_out = pd.DataFrame(Q3_out, columns = ['sub', 'r2_score'])

lm_PIG.coef_

temp = Q3[Q3.SUBJECT == 'POULTRY']
lm_POULTRY.predict(temp[['TIME']])
    
#%%

# =============================================================================
# =============================================================================
# # 문제 05 유형(DataSet_05.csv 이용)
#
# 구분자 : comma(“,”), 8,068 Rows, 12 Columns, UTF-8 인코딩
#
# A자동차 회사는 신규 진입하는 시장에 기존 모델을 판매하기 위한 마케팅 전략을 
# 세우려고 한다. 기존 시장과 고객 특성이 유사하다는 전제 하에 기존 고객을 세분화하여
# 각 그룹의 특징을 파악하고, 이를 이용하여 신규 진입 시장의 마케팅 계획을 
# 수립하고자 한다. 다음은 기존 시장 고객에 대한 데이터이다.
#

# 컬 럼 / 정 의 / Type
# ID / 고유 식별자 / Double
# Age / 나이 / Double
# Age_gr / 나이 그룹 (10/20/30/40/50/60/70) / Double
# Gender / 성별 (여성 : 0 / 남성 : 1) / Double
# Work_Experience / 취업 연수 (0 ~ 14) / Double
# Family_Size / 가족 규모 (1 ~ 9) / Double
# Ever_Married / 결혼 여부 (Unknown : 0 / No : 1 / Yes : 2) / Double
# Graduated / 재학 중인지 여부 / Double
# Profession / 직업 (Unknown : 0 / Artist ~ Marketing 등 9개) / Double
# Spending_Score / 소비 점수 (Average : 0 / High : 1 / Low : 2) / Double
# Var_1 / 내용이 알려지지 않은 고객 분류 코드 (0 ~ 7) / Double
# Segmentation / 고객 세분화 결과 (A ~ D) / String
# =============================================================================
# =============================================================================


#(참고)
#1
# import pandas as pd
# #2
# from scipy.stats import chi2_contingency
# #3
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
# import pydot

#%%

import pandas as pd
import numpy as np

dataset5 = pd.read_csv('./Dataset/DataSet_05.csv', na_values = ['NA', '?', '', ' '])
dataset5.columns
dataset5.dtypes
dataset5.shape
dataset5.info()


#%%

# =============================================================================
# 1.위의 표에 표시된 데이터 타입에 맞도록 전처리를 수행하였을 때, 데이터 파일 내에
# 존재하는 결측값은 모두 몇 개인가? 숫자형 데이터와 문자열 데이터의 결측값을
# 모두 더하여 답하시오.
# (String 타입 변수의 경우 White Space(Blank)를 결측으로 처리한다) (답안 예시) 123
# =============================================================================

dataset5.isna().sum().sum()

# 답
# 1166



#%%

# =============================================================================
# 2.이어지는 분석을 위해 결측값을 모두 삭제한다. 그리고, 성별이 세분화(Segmentation)에
# 영향을 미치는지 독립성 검정을 수행한다. 수행 결과, p-value를 반올림하여 소수점
# 넷째 자리까지 쓰고, 귀무가설을 기각하면 Y로, 기각할 수 없으면 N으로 기술하시오. 
# (답안 예시) 0.2345, N
# =============================================================================

Q2 = dataset5.dropna()

len(dataset5)
len(Q2)

# 범주형 끼리의 독립성 검정은 카이제곱검정
# 카이제곱 검정은 빈도표가 필요

Q2_tab = pd.crosstab(index=Q2.Gender,
                     columns=Q2.Segmentation)

from scipy.stats import chi2_contingency

Q2_out = chi2_contingency(Q2_tab)

# 답
Q2_out[1] # 0.0031
# H0 : 두 변수는 연관이 없다
# H1 : 연관이 있다
# pvalue가 유의수준 0.05보다 작으므로 귀무가설을 기각
# Y


#%%

# =============================================================================
# 3.Segmentation 값이 A 또는 D인 데이터만 사용하여 의사결정 나무 기법으로 분류
# 정확도를
# 측정해 본다. 
# - 결측치가 포함된 행은 제거한 후 진행하시오.
# - Train대 Test 7대3으로 데이터를 분리한다. (Seed = 123)
# - Train 데이터를 사용하여 의사결정나무 학습을 수행하고, Test 데이터로 평가를
# 수행한다.
# - 의사결정나무 학습 시, 다음과 같이 설정하시오:
# • Feature: Age_gr, Gender, Work_Experience, Family_Size, 
#             Ever_Married, Graduated, Spending_Score
# • Label : Segmentation
# • Parameter : Gini / Max Depth = 7 / Seed = 123
# 이 때 전체 정확도(Accuracy)를 소수점 셋째 자리 이하는 버리고 소수점 둘째자리까지
# 기술하시오.
# (답안 예시) 0.12
# =============================================================================

Q3 = Q2[Q2.Segmentation.isin(['A','D'])]

Q3.isna().sum()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

train, test = train_test_split(Q3,
                               test_size = 0.3,
                               random_state = 123)
train.columns
x_var= ['Age_gr', 'Gender', 'Work_Experience', 'Family_Size',
       'Ever_Married', 'Graduated', 'Spending_Score']


dt = DecisionTreeClassifier(max_depth=7,
                            random_state=123)

dt.fit(train[x_var], train.Segmentation)
pred = dt.predict(test[x_var])

# 답
dt.score(test[x_var], test.Segmentation)
# 0.68