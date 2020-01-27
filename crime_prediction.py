# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:54:09 2020

@author: user
"""
import numpy as np
import crimepkg.helpers as hp
import importlib

importlib.reload(hp)


helper = hp.DenverHelper('denver.csv','demographic.csv')

print("Section 1. 미국 덴버 - 범죄 데이터 시각화")
print("")

print("## 초기 데이터 미리보기 ##")
print(helper.getCrimes().head(5))
print("")


print("## 범죄유형 ##")
crime_categories = helper.getCategorys()
print(crime_categories)
print("")

print("## 범죄유형데이터 문자에서 숫자범위(OFFENSE_CATEGORY_ID -> Crime_Type)로 변경 ##")
helper.changeCategorytoNum()
print(helper.getCrimes())
print("")

print("## 범죄유형(1~6)별 사건 개수 ##")
print(helper.getCountByCategorys())
print("")

print("## 카운터 초기화 및 Date관련 포맷팅하여 컬럼 수정(추가,삭제)##")
print("* 카운터 : 유형별,년도별,달별,일별,시간별 범죄발생 수를 저장하고있는 변수)")
helper.initCount_AddDateColumns()
print("작업완료")
print("")

print("## Year 카운터 출력 ##")
print(helper.getCountByYears())
print("")

print("## Month 카운터 출력 ##")
print(helper.getCountByMonths())
print("")

print("## Day 카운터 출력 ##")
print(helper.getCountByDays())
print("")

print("## Hour 카운터 출력 ##")
print(helper.getCountByHours())
print("")

print("## Type 카운터 출력 ##")
print(helper.getCountByCategorys())
print("")

print("## 컬럼 변경된 범죄 데이터 출력(FIRST_OCCURRENCE_DATE삭제,Date관련 컬럼들 추가)")
print(helper.getCrimes().head(5))
print("범죄데이터 개수(사건수):",len(helper.getCrimes()))
print("")



#그래프 그리기
# arr =[bar,steam,xticks,label,title]
# def graph(self,arr,count,xtick,label,title,color):
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
time_range = ['1:00AM - 4:59AM','5:00AM - 8:59AM','9:00AM - 12:59PM','1:00PM - 4:59PM','5:00PM - 8:59PM','9:00PM - 00:59AM']
years = ['2015','2016','2017','2018','2019','2020']

print("## 범죄유형별 사건수 백분율 그래프 그리기 ##")
helper.graph([0,1,0,1,1],helper.getCountByCategorys(),None,'Category Type','Percentage of a particular category type',None)
print("")

print("## 년도별 사건수 백분율 그래프 그리기 ##")
helper.graph([1,0,1,1,1],helper.getCountByYears(),years,'Year','Percentage of a crime occurence on a particular year','red')
print("")


print("## 월별 사건수 백분율 그래프 그리기 ##")
helper.graph([1,0,1,1,1],helper.getCountByMonths(),months,'Month','Percentage of a crime occurence in a particular month','blue')
print("")


print("## 일별 사건수 백분율 그래프 그리기 ##")
helper.graph([1,0,1,1,1],helper.getCountByDays(),days,'Day','Percentage of a crime occurence in a particular day','green')
print("")


print("## 시간별 사건수 백분율 그래프 그리기 ##")
helper.graph([1,0,1,1,1],helper.getCountByHours(),time_range,'Time Range','Percentage of a crime occurence in a particular time','yellow')
print("")

print("## 지역별 사건수 백분율 그래프 그리기 ##")
helper.graph([0,1,0,1,1],helper.getCountByNeighbours(),None,'Neighbourhood ID','Percentage of a crime occurence at a particular place',None)
print("")

print("## 시간 and 범죄유형별 백분율 그래프 그리기 ##")
a = ['1:00AM - 4:59AM','5:00AM - 8:59AM','9:00AM - 12:59PM','1:00PM - 4:59PM','5:00PM - 8:59PM','9:00PM - 00:59AM']
#범죄유형과 시간범주를 축으로 축으로 사건 개수를 가지고있는 2차원 행렬 가져오기
type_time_matrix=helper.getTypeTimeMatrix()
helper.type_graph(type_time_matrix,len(a),a,'Crime type distribution in Denver based on time')


print("## 년도 and 범죄유형별 백분율 그래프 그리기 ##")
a = ['2015','2016','2017','2018','2019','2020']
#범죄유형과 년도값을 축으로 사건 개수를 가지고있는 2차원 행렬 가져오기
type_year_matrix = helper.getTypeYearMatrix()
helper.type_graph(type_year_matrix,len(a),a,'Crime type distribution in Denver based on years')
print("")

print("Section 2. 범죄 유형 예측")
print("")

#Crime_Location컬럼 마지막으로 보내기
helper.moveColumntoLast('Crime_Location')

crimes = helper.getCrimes()

#범죄데이터 5조각으로 쪼개기
set1 = crimes.iloc[0:72463]
set2 = crimes.iloc[72463:144926]
set3 = crimes.iloc[144926:217389]
set4 = crimes.iloc[217389:289852]
set5 = crimes.iloc[289852:]


#각각은 모델별로 정확도를 저장하는 배열
#set1,2,3,4를 트레이닝 셋,5를 테스트셋으로 모델별 정확도를 측정하고,
#2,3,4,5를 트레이닝 셋 1을 테스트를 셋으로 모델별 정확도를 측정하고 이런식으로
#돌려가면서 총 5번의 회전을 통해 모델별로 5개의 정확도를 얻을수 있다.
#이 정확도들의 평균을 구하여 모델의 정확도를 결정한다.
#단순나이브베이즈모델 정확도 리스트
simple_accuracy_list = []
#다항분포 나이브베이즈 모델 정확도 리스트
MultinomialNB_accuracy_list = []
#가우시안 나이브베이즈 모델 정확도 리스트
GaussianNB_accuracy_list = []
#베르누이분포 나이브베이즈 모델 정확도 리스트
BernoulliNB_accracy_list = []

#트레이닝 셋과 테스트셋을 4:1 비율로 쪼갠다.
train_set,test_set = helper.setTrainTestData(set1,set2,set3,set4,set5)

#result:test_set을 기반으로 각 사건별(월,일,시간,지역)로 가장 발생할 확률이 높은 범죄유형을 
#test_set_type: 사건별 실제 범죄유형
result,test_set_type = helper.simpleBeysianTest(train_set,test_set)
mulresult,mul_test_set_type = helper.MultinomialNBTest(train_set,test_set)
gausresult,gaus_test_set_type = helper.GaussianNBTest(train_set,test_set)
bernresult,bern_test_set_type = helper.BernoulliNBTest(train_set,test_set)

#각 모델의 정확도 측정
simple_accracy = helper.calculate_accuracy(result,test_set_type)
MultinomialNB_accuracy = helper.calculate_accuracy(mulresult,mul_test_set_type)
GaussianNB_accuracy = helper.calculate_accuracy(gausresult,gaus_test_set_type)
BernoulliNB_accracy = helper.calculate_accuracy(bernresult,bern_test_set_type)

#정확도 저장
simple_accuracy_list.append(simple_accracy)
MultinomialNB_accuracy_list.append(MultinomialNB_accuracy)
GaussianNB_accuracy_list.append(GaussianNB_accuracy)
BernoulliNB_accracy_list.append(BernoulliNB_accracy)


#P(type|month,day,time,location)
#특정 월,일,시간,지역을 기반으로 가장 확률이 높은 범죄유형을 예측
#[11,5,5,7] => [month,day,time,location]
predict_val = helper.simpleBeysianPredict(train_set,[11,5,5,7])
print("## Crime Type Prediction ##")
print(" Simple Beysian [11,5,5,7] => [month,day,time,location] 범죄유형 예측")
print(predict_val)
print(" Simple Beysian 1차 테스트 정확도")
print(simple_accracy)

train_set,test_set = helper.setTrainTestData(set2,set3,set4,set5,set1)
result,test_set_type = helper.simpleBeysianTest(train_set,test_set)
mulresult,mul_test_set_type = helper.MultinomialNBTest(train_set,test_set)
gausresult,gaus_test_set_type = helper.GaussianNBTest(train_set,test_set)
bernresult,bern_test_set_type = helper.BernoulliNBTest(train_set,test_set)

simple_accracy = helper.calculate_accuracy(result,test_set_type)
MultinomialNB_accuracy = helper.calculate_accuracy(mulresult,mul_test_set_type)
GaussianNB_accuracy = helper.calculate_accuracy(gausresult,gaus_test_set_type)
BernoulliNB_accracy = helper.calculate_accuracy(bernresult,bern_test_set_type)

simple_accuracy_list.append(simple_accracy)
MultinomialNB_accuracy_list.append(MultinomialNB_accuracy)
GaussianNB_accuracy_list.append(GaussianNB_accuracy)
BernoulliNB_accracy_list.append(BernoulliNB_accracy)

print(" Simple Beysian 2차 테스트 정확도")
print(simple_accracy)

train_set,test_set = helper.setTrainTestData(set3,set4,set5,set1,set2)
result,test_set_type = helper.simpleBeysianTest(train_set,test_set)
mulresult,mul_test_set_type = helper.MultinomialNBTest(train_set,test_set)
gausresult,gaus_test_set_type = helper.GaussianNBTest(train_set,test_set)
bernresult,bern_test_set_type = helper.BernoulliNBTest(train_set,test_set)

simple_accracy = helper.calculate_accuracy(result,test_set_type)
MultinomialNB_accuracy = helper.calculate_accuracy(mulresult,mul_test_set_type)
GaussianNB_accuracy = helper.calculate_accuracy(gausresult,gaus_test_set_type)
BernoulliNB_accracy = helper.calculate_accuracy(bernresult,bern_test_set_type)

simple_accuracy_list.append(simple_accracy)
MultinomialNB_accuracy_list.append(MultinomialNB_accuracy)
GaussianNB_accuracy_list.append(GaussianNB_accuracy)
BernoulliNB_accracy_list.append(BernoulliNB_accracy)

print(" Simple Beysian 3차 테스트 정확도")
print(simple_accracy)

train_set,test_set = helper.setTrainTestData(set4,set5,set1,set2,set3)
result,test_set_type = helper.simpleBeysianTest(train_set,test_set)
mulresult,mul_test_set_type = helper.MultinomialNBTest(train_set,test_set)
gausresult,gaus_test_set_type = helper.GaussianNBTest(train_set,test_set)
bernresult,bern_test_set_type = helper.BernoulliNBTest(train_set,test_set)

simple_accracy = helper.calculate_accuracy(result,test_set_type)
MultinomialNB_accuracy = helper.calculate_accuracy(mulresult,mul_test_set_type)
GaussianNB_accuracy = helper.calculate_accuracy(gausresult,gaus_test_set_type)
BernoulliNB_accracy = helper.calculate_accuracy(bernresult,bern_test_set_type)

simple_accuracy_list.append(simple_accracy)
MultinomialNB_accuracy_list.append(MultinomialNB_accuracy)
GaussianNB_accuracy_list.append(GaussianNB_accuracy)
BernoulliNB_accracy_list.append(BernoulliNB_accracy)

print(" Simple Beysian 4차 테스트 정확도")
print(simple_accracy)

train_set,test_set = helper.setTrainTestData(set5,set1,set2,set3,set4)
result,test_set_type = helper.simpleBeysianTest(train_set,test_set)
mulresult,mul_test_set_type = helper.MultinomialNBTest(train_set,test_set)
gausresult,gaus_test_set_type = helper.GaussianNBTest(train_set,test_set)
bernresult,bern_test_set_type = helper.BernoulliNBTest(train_set,test_set)

simple_accracy = helper.calculate_accuracy(result,test_set_type)
MultinomialNB_accuracy = helper.calculate_accuracy(mulresult,mul_test_set_type)
GaussianNB_accuracy = helper.calculate_accuracy(gausresult,gaus_test_set_type)
BernoulliNB_accracy = helper.calculate_accuracy(bernresult,bern_test_set_type)

simple_accuracy_list.append(simple_accracy)
MultinomialNB_accuracy_list.append(MultinomialNB_accuracy)
GaussianNB_accuracy_list.append(GaussianNB_accuracy)
BernoulliNB_accracy_list.append(BernoulliNB_accracy)

print(" Simple Beysian 5차 테스트 정확도")
print(simple_accracy)

print("### 모델별 정확도 ###")
print(round(np.mean(simple_accuracy_list),2) , round(np.mean(MultinomialNB_accuracy_list),2) , round(np.mean(GaussianNB_accuracy_list),2) , round(np.mean(BernoulliNB_accracy_list),2))




##### Section3 . Predicting crime occurence #####
### 여기부터는 전혀 모르겠어서 냅뒀음.. 모듈화 작업 진행중.. ###
print("Section 3. 시계열데이터기반 미래 범죄 발생 수 예측")


################ 모듈화 작업 진행중 ########################
############### 밑에 주석 풀면 코드실행에 문제 생김 ############
'''
c = np.zeros((12,6))
for j in range(12):
    for i in range(len(months)):
        if(months[i]==j+1):
            c[j][years[i]-1] = c[j][years[i]-1] + 1
c = np.transpose(c)



for i in range(6):
    plt.plot(c[i][:])
    plt.title('Year '+ str(i+2015))
    plt.show()
plt.plot(c.flatten())
plt.show()


# Time Series Plots :
# 1. The first 6 graphs show the trend of number of crimes during the span of year.
# 2. The last graph shows number of crimes occured during every month, year pair that is combining all the above graphs.
# 3. We can observe that in year 2013, 2014 , 2016 and 2017 maximum crime occurs during 3rd quater of the year.
# 4. We can observe that in year 2015 and 2018 maximum crime occurs during 2nd quater of the year.

# #### Calculating Rolling Mean

# In[ ]:


def add_months(sourcedate,months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

t = c.flatten()
ans = datetime.date(2015, 1,1)
table = pd.DataFrame(columns=['Date','Value'])


for i in range(len(t)):
    table.loc[i] = [ans,t[i]]
    ans = add_months(ans,1)
table['Date'] = pd.to_datetime(table['Date'],infer_datetime_format=True)
indexedDataset = table.set_index(['Date'])


# In[ ]:


rolmean = indexedDataset.rolling(window=12).mean()
rolstd = indexedDataset.rolling(window=12).std()

plt.plot(indexedDataset['Value'],color='green')
plt.show()
plt.plot(rolmean,color='red')
plt.show()
plt.plot(rolstd,color='blue')
plt.show()


# Inferences
# 1. We observe that rolling mean drastically changes during the 3rd and 4th quarter of time line series.
# 2. Moereover the value of rolling mean lies between 2000 to 6000 range.
# 3. We observe a sudden decrease in crime at the end of year 2017 and beginning of the year 2018.
# 4. Standard deviation drastically changes at the end of year 2017. This is observed because of sudden change in data values to 0 in that time period.
# 
# We can infer from above graphs that some kind of anomaly was observed in noting the crimes and storing their values.

# In[1]:


indexedDataset_log = np.log(indexedDataset)

rolmean = indexedDataset_log.rolling(window=12).mean()
rolstd = indexedDataset_log.rolling(window=12).std()

plt.plot(indexedDataset_log['Value'],color='green')
plt.show()
plt.plot(rolmean,color='red')
plt.show()
plt.plot(rolstd,color='blue')
plt.show()


# In[ ]:


indexedDataset_cbrt = np.cbrt(indexedDataset)

rolmean = indexedDataset_cbrt.rolling(window=12).mean()
rolstd = indexedDataset_cbrt.rolling(window=12).std()

plt.plot(indexedDataset_cbrt['Value'],color='green')
plt.show()
plt.plot(rolmean,color='red')
plt.show()
plt.plot(rolstd,color='blue')
plt.show()


# In[ ]:


indexedDataset_pow = np.power(indexedDataset,1/10)

rolmean = indexedDataset_pow.rolling(window=12).mean()
rolstd = indexedDataset_pow.rolling(window=12).std()

plt.plot(indexedDataset_pow['Value'],color='green')
plt.show()
plt.plot(rolmean,color='red')
plt.show()
plt.plot(rolstd,color='blue')
plt.show()


# In[ ]:


indexedDataset_pow = np.power(indexedDataset,1/20)

rolmean = indexedDataset_pow.rolling(window=12).mean()
rolstd = indexedDataset_pow.rolling(window=12).std()

plt.plot(indexedDataset_pow['Value'],color='green')
plt.show()
plt.plot(rolmean,color='red')
plt.show()
plt.plot(rolstd,color='blue')
plt.show()


# In[ ]:


autocorrelation_plot(indexedDataset)
plt.show()


# Knowing the autocorrelation values :
# 1. We find that values observes are mostly negative and sometimes tend to go to 0. As observed in the last part of the data plot

# In[ ]:

# Please change indexedDataset to float64
indexedDataset = indexedDataset.astype('float64')
print(indexedDataset)
model = ARIMA(indexedDataset, order=(10,1,0))

model_fit = model.fit()
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
'''







