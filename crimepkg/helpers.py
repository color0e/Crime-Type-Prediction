# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:29:54 2020

@author: user
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

class DenverHelper:
    #초기화 범죄데이터와 지역데이터를 받아 초기화한다.
    def __init__(self,crimes_file,neighbours_file):
        self.num_days = 7 #7일
        self.num_months = 12 #12달
        self.num_time_range = 6 
        self.num_n_d = 78 
        self.num_years = 6 # 2015-2020
        self.num_type = 6
        self.CRIME_TYPE = 'OFFENSE_CATEGORY_ID'
        self.DATE = 'FIRST_OCCURRENCE_DATE'
        
        crimes = pd.read_csv(crimes_file)
        crimes = crimes[(crimes['IS_CRIME'] != 0)] 
        crimes = crimes[['INCIDENT_ID',self.CRIME_TYPE, 'FIRST_OCCURRENCE_DATE',
        'NEIGHBORHOOD_ID', 'IS_CRIME']]
        self.crimes = crimes
        
        neighbours = pd.read_csv(neighbours_file)
        
        #지역정보 데이터에서 지역이름을 범죄데이터에 있는 지역이름 포맷과 동일하게 변경
        current = neighbours[['NBHD_ID', 'NBRHD_NAME']]
        current = current.sort_values(by=[current.columns[0]])
        n_ref = current[current.columns[1]].values
        for i in range(len(n_ref)):
            x = n_ref[i]
            x = str(x)
            x = x.lower()
            y = ""
            f = 1
            for j in range(len(x)):
                if(x[j]=='/' and x[j+1]==' '):
                    f=0  
                elif(x[j]==' ' and f==1):
                    y = y + '-'
                    f = 0
                elif(x[j]==' ' and f==0):
                    continue
                elif(x[j]=='-'):
                    continue
                else:
                    y = y + x[j]
                    f = 1
            n_ref[i] = y
        
        #범죄데이터에서 NEIGHBORHOOD_ID컬럼삭제하고 Crime_Location컬추가(지역을 숫자로 표기한 데이터)
        ans = []
        self.n_count_d = np.zeros(78)
        neighbours = self.crimes['NEIGHBORHOOD_ID'].values
        for i in range(len(neighbours)):
            x = neighbours[i]
            x = str(x)
            for j in range(len(n_ref)):
                if(x==n_ref[j]):
                    ans.append(j+1)
                    self.n_count_d[j] = self.n_count_d[j] + 1

        self.crimes['Crime_Location'] = ans
        self.crimes = self.crimes.drop('NEIGHBORHOOD_ID',axis=1)

        self.neighbours = neighbours
    
    #범죄 데이터들 리턴
    def getCrimes(self):
        return self.crimes
    
    #범죄 데이터유형들 리턴(사건수가 아님 그냥 유형개수만큼 리턴)
    def getCategorys(self):
        return pd.unique(self.crimes[self.CRIME_TYPE].values.ravel()).tolist()
    
    #범죄 유형으로 모든 사건들 출력
    def getCategoryData(self):
        return self.crimes[self.CRIME_TYPE].values
    
    #월 값으로 모든 사건들 출력
    def getMonthData(self):
        return self.crimes['Crime_Month'].values
    
    #년도 값으로 모든 사건들 출력
    def getYearData(self):
        return self.crimes['Crime_Year'].values
    
    #시간범위 값으로 모든 사건들 출력
    def getTimeData(self):
        return self.crimes['Crime_Time'].values
    
    #일 값으로 모든 사건들 출력
    def getDayData(self):
        return self.crimes['Crime_Day'].values
    
    #문자 범죄유형컬럼을 삭제하고 숫자 범죄유형컬럼을 추가하는 메서드
    def changeCategorytoNum(self):
        self.c_count_d = np.zeros(self.num_type)
        categoryData = self.getCategoryData()
        
        for i in range(len(categoryData)):
            categoryData[i] = self.transform(categoryData[i])
            self.c_count_d[categoryData[i]-1] = self.c_count_d[categoryData[i]-1]+1
        
        self.crimes['Crime_Type'] = categoryData
        
        self.crimes = self.crimes.drop([self.CRIME_TYPE],axis=1)
        self.CRIME_TYPE = 'Crime_Type'
        
    #카운터(유형별,년도별,달별,일별,시간별 범죄발생 수를 저장하고있는 변수) 초기화 
    #및 범죄데이터에 연도,달,일,시간 컬럼 추가하고  FIRST_OCCURRENCE_DATE컬럼을 날리는 메서드     
    def initCount_AddDateColumns(self):
        months = []
        days = []
        hours = []
        years = []
        
        self.d_count_d = np.zeros(self.num_days) #[0,0,0,0,0,0,0] 7개
        self.m_count_d = np.zeros(self.num_months) #12
        self.h_count_d = np.zeros(self.num_time_range) #6
        self.y_count_d = np.zeros(self.num_years) #5
        
        date = self.crimes[self.DATE].values
        
        for i in range(len(date)):
            x = date[i]
            x = str(x)
            
            month, day, year , hour , minutes , seconds , f = self.convert(x)  
            years.append(year-2015)#0-5
            months.append(month)#1-12
            ans = datetime.date(year, month, day)
            #ans = 2019-12-31
            t = ans.weekday()
            #t 월 =0 화=1...일=6
            days.append(t+1)
            #t = 1~7 ex) 월=1
            j = self.conversion(hour,f)
            #시간 범위 출력 1~6
            hours.append(j)
            
            #카운트 초기화
            self.m_count_d[month-1] = self.m_count_d[month-1]+1
            self.d_count_d[t] = self.d_count_d[t]+1
            self.h_count_d[j-1] = self.h_count_d[j-1] + 1
            self.y_count_d[year-2015] = self.y_count_d[year-2015] + 1
        
        #컬럼 추가     
        self.crimes['Crime_Month'] = months #1~12
        self.crimes['Crime_Day'] = days #1~7
        self.crimes['Crime_Time'] = hours #1~6
        self.crimes['Crime_Year'] = years #0~5
        
        #컬럼 삭제
        self.crimes = self.crimes.drop([self.DATE],axis=1)
        
    
    # 범죄유형별 발생한 사건의 수를 리턴함
    def getCountByCategorys(self):
        return self.c_count_d
    
    # 년도별 발생한 사건의 수를 리턴함
    def getCountByYears(self):
        return self.y_count_d
        
    # 월별 발생한 사건의 수를 리턴함
    def getCountByMonths(self):
        return self.m_count_d
    
    # 일별 발생한 사건의 수를 리턴함
    def getCountByDays(self):
        return self.d_count_d
    
    # 시간범위별 발생한 사건의 수를 리턴함
    def getCountByHours(self):
        return self.h_count_d
    
    # 지역별 발생한 사건의 수를 리턴함
    def getCountByNeighbours(self):
        return self.n_count_d
        
    #범죄유형을 숫자로 변환하는 함수 - in transform function
    def func(self,s):
        s = s.lower()
        if(s=='assault' or s=='murder'):
            return 1
        if(s=='drug' or s=='alcohol' or s=='drugs' or s=='drunk' or s=='roll'):
            return 2
        if(s=='other'):
            return 3
        if(s=='pimping' or s=='initmate' or s=='public' or s=='disorder' or s=='sexual' or s=='indecent' or s=='bigamy' or s=='lewd' or s=='sex' or s=='pandering'):
            return 4
        if(s=='purse' or s=='prowlwer' or s=='pickpocket' or s=='till' or s=='theft' or s=='shoplifting' or s=='burglary' or s=='larceny' or s=='robbery' or s=='stolen'):
           return 5
        if(s=='counterfeit' or s=='identity' or s=='bunco' or s=='white' or s=='collar' or s=='embezzlement' or s=='credit' or s=='extortion' or s=='bribery' or s=='employee'):
           return 6
        return 3
   
    #범죄유형을 숫자데이터로 변환하는 함수
    def transform(self,s):
        t = s.split('-')
        for x in t:
            j = self.func(x)
            if(j!=3):
                return j
        return 3
    
    # 입려된 데이터를 포맷팅하여 month,day,year,hour,minutes,seconds,f(f=1 : AM , f=0 : PM) 값을 리턴함.
    # 입력데이터 예)6/15/2016 11:31:00 PM
    def convert(self,x):
        y = ""
        f = 0
        for j in range(len(x)):
            if(x[j]==' ' or x[j]==':'):
                if(x[j+1]=='A' or x[j+1]=='P'):
                    if(x[j+1]=='A'):
                        f=1
                    break
                else:
                    y= y + '/'      
            else:
                y = y + x[j]
        month, day, year , hour , minutes , seconds = (int(t) for t in y.split('/')) 
        return month, day, year , hour , minutes , seconds , f
    
    #시간범위를 숫자(1~6)로 리턴한다. 
    #1=>'1:00AM - 4:59AM', 2=>'5:00AM - 8:59AM',3=>'9:00AM - 12:59PM',4=>'1:00PM - 4:59PM',5=>'5:00PM - 8:59PM',6=>'9:00PM - 00:59AM'
    def conversion(self,hour,f):
        if(f==1):
            if(hour>=1 and hour<=4):
                return 1
            elif(hour>=5 and hour<=8):
                return 2 
            elif(hour>=9 and hour<=11):
                return 3
            else:
                return 6
        else:
            if(hour==12):
                return 3
            elif(hour>=1 and hour<=4):
                return 4
            elif(hour>=5 and hour<=8):
                return 5
            else:
                return 6
    
    #배열의 각 요소들을 전체합으로 각각 나눈다.
    def prob(self,arr):
        s = sum(arr)
        return [(x/s) for x in arr]
    
    #배열 값들을 백분율로 변환
    def get_percentage(self,arr):
        arr = self.prob(arr)
        return [x*100 for x in arr]
    
    #2차원 배열용 행기준 백분율 변환(열들의 값들을 모두 더하여 각 요소 백분율)
    #leng 범죄유형말고 다른 데이터의 범주개수
    def get_percentage_two(self,arr,leng):
        for i in range(leng):
            s = 0
            for j in range(self.num_type):
                s = s + arr[i][j]
            for j in range(self.num_type):
                arr[i][j] = arr[i][j] / s
        return arr
    
    #그래프 그리기 arr =[bar,steam,xticks,clabel,title]
    def graph(self,arr,count,xtick,label,title,rgb):
        #카운터 백분율로 변환
        per_d = self.get_percentage(count)
        if(arr[0]==1):
            bar_width = 0.5
            plt.bar((np.arange(len(per_d))+1)*2,per_d,bar_width,color=rgb,label='Denver')
        if(arr[1]==1):
            plt.stem(np.arange(len(per_d))+1,per_d,use_line_collection=True)
        if(arr[2]==1):
            plt.xticks((np.arange(len(xtick))+1)*2,xtick, fontsize=10,rotation=30)
        if(arr[3]==1):
            plt.xlabel(label)
        if(arr[4]==1):
            plt.title(title)
            
        plt.ylabel('Percentage of occurences')
        plt.show()
        
    def getTypeTimeMatrix(self):
        times = self.getTimeData()
        types = self.getCategoryData()
        
        time_type = np.zeros((6,6))
        for i in range(len(times)):
            b = times[i] - 1
            a = types[i] - 1
            time_type[b][a] = time_type[b][a] + 1 
        
        return time_type
    
    def getTypeYearMatrix(self):
        years = self.getYearData()
        types = self.getCategoryData()
        
        year_type = np.zeros((6,6))
        for i in range(len(years)):
            b = years[i] - 1
            a = types[i] - 1
            year_type[b][a] = year_type[b][a] + 1 
        
        return year_type
    
    def type_graph(self,day_type,leng,xtick,title):
        plt.title(title)
        day_type = np.transpose(self.get_percentage_two(day_type,leng) * 100)
        bar_width = 0.75 
        plt.bar((np.arange(len(day_type[0]))*7),day_type[0],bar_width,color='blue',label='Assault')
        plt.bar(((np.arange(len(day_type[1]))*7)+1),day_type[1],bar_width,color='red',label='Drug Alcohol')
        plt.bar(((np.arange(len(day_type[2]))*7)+2),day_type[2],bar_width,color='green',label='Other Crime')
        plt.bar(((np.arange(len(day_type[3]))*7)+3),day_type[3],bar_width,color='black',label='Public Disorder')
        plt.bar(((np.arange(len(day_type[4]))*7)+4),day_type[4],bar_width,color='yellow',label='Theft')
        plt.bar(((np.arange(len(day_type[5]))*7)+5),day_type[5],bar_width,color='brown',label='White Collar Crime')
        plt.xticks((np.arange(len(day_type[0]))*7)+3,xtick, fontsize=10, rotation=30)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('Percentage')
        plt.show()

    def moveColumntoLast(self,col):
        temp = self.crimes[col]
        self.crimes.drop(columns=[col], inplace=True)
        self.crimes[col]=temp
    
    #베이지안 공식
    def bayesian_formula(self,a,b,c):
        for i in range(6):
            for j in range(c):
                a[i][j] = a[i][j] * b[i]
        return a
    
    #b를 기준으로 범죄유형 개수를 백분율로 변환 ex)b가 월이라면 1월에 1번유형은 0.3 ,1월에 2번유형은 0.2,1월에 3번 유형 0.1... 다합하면 1
    #ex)b가 월이라면 월별로 가장 많이 발생한 범죄유형을 알수있음
    #ex)a는 월별 각 범죄유형들의 수
    def type_prob(self,a,b):
        for i in range(6):
            s = 0
            for j in range(b):
                s = s + a[i][j]
            for j in range(b):
                a[i][j] = a[i][j] / s
        return a
    
    #트레이닝셋과 테스트셋을 4:1로 쪼갬
    def setTrainTestData(self,a,b,c,d,e):
        return pd.concat([a,b,c,d]),e
    
    def find_matrix(self,a,b,prob_c_d,num):
        t = np.zeros((num,6))
        for i in range(len(a)):
            #특정월 and 특정유형을 축으로(2차원배열) 수세기 (t=>행:월,열:범죄유형)
            t[a[i]-1][b[i]-1] = t[a[i]-1][b[i]-1] + 1
            
        #transpose 행열 바꾸기 여기서는 범죄유형을 행으로 위치
        t = np.transpose(t)
        
        #각 범죄유형 월별 비율화 ex ) 1월 도둑질(범죄유형)의 수/발생한 도둑질(범죄유형)의수 
        t = self.type_prob(t,num)
        
        # 해당함수는 특정월에 특정범죄가 발생할 확률을 각각 구하여 저장하는 것으로 보여진다.
        # ex) 1월에 도둑질이 발생할 확률(근사치) P(type(Theft)|month(1)) = P(month(1)|type(Theft))*P(type(Theft)) ??
        # 위의 ex는 솔직히 잘모름.. 그냥 이렇게 썼을거 같다고 추측한 공식
        # 어쨌든 최종적으로 P(type|month,day,time,location)을 구하기 위해 필요한 식 요소를
        # bayesian_formula함수를 통해 구했을것으로 추측되어짐
        t = self.bayesian_formula(t,prob_c_d,num)
        return t

    def count(self,a,b):
        ty = np.zeros(b)
        for i in range(len(a)):
            ty[int(a[i])-1] = ty[int(a[i])-1] + 1
        return ty


    def calculate_accuracy(self,result,test_out):
        c =0 
        for i in range(len(result)):
            if(result[i]==test_out[i]):
                c = c + 1
                
        #최종적으로 확률을 기반으로 추측한 테스트 데이터의 범죄유형과 테스트케이스의 실제 범죄유형을 비교하여 정확도를 계산한다.
        return round(c/len(result),2)
    
    
    def simpleBeysianTest(self,train_set,test_set):
            #트레이닝셋 - 각 사건별 월정보
            train_set_month = train_set['Crime_Month'].values
            #트레이닝셋 - 각 사건별 지역정보
            train_set_location = train_set['Crime_Location'].values
            #트레이닝셋 - 각 사건별 범죄유형정보
            train_set_type = train_set['Crime_Type'].values
            #트레이닝셋 - 각 사건별 일자정보
            train_set_day = train_set['Crime_Day'].values
            #트레이닝셋 - 각 사건별 시간정보
            train_set_time = train_set['Crime_Time'].values
           
            #태스트셋 - 각 사건별 월정보
            test_set_month = test_set['Crime_Month'].values
            #태스트셋 - 각 사건별 월정보
            test_set_location = test_set['Crime_Location'].values
            #태스트셋 - 각 사건별 월정보
            test_set_type = test_set['Crime_Type'].values
            #태스트셋 - 각 사건별 월정보
            test_set_day = test_set['Crime_Day'].values
            #태스트셋 - 각 사건별 월정보
            test_set_time = test_set['Crime_Time'].values
            #태스트셋 - 각 사건별 월정보 <= 안쓰임... 월,일,시간을 가지고는 미래확률을 계산할수있으나 year가 포함되면 활용할수가없음..
            test_set_year = test_set['Crime_Year'].values
            
            #count function : month 데이터를 매개변수로 넣을경우 월별 범죄발생 수 배열 리턴 (1차원배열)
            #prob function : 월별 범죄발생수를 매개변수로 넣을경우, 각 월별로 범죄발생수를 비율화함 [0.1,0.05,....0.15] 다더하면 1
            #prob_m_d 어떤 월이 가장 범죄가 많이발생했는지 확인가능 , 월별 범죄발생의 비율
            prob_m_d = self.prob(self.count(train_set_month,12)) #P(m) ?
            #prob_c_d 어떤 유형의 범죄가 가장많이 발생했는지 비율화
            prob_c_d = self.prob(self.count(train_set_type,6))
            
            #(month,day,time,n,y)_type_d => P(type|month,day,time,location)을 구하기 위해 필요한 식 요소라고 추측됨
            #즉 특정월,일,시간,지역에서 발생할 가능성이 가장 높은 범죄유형을 확률계산을 통해 가장확률이 높은것으로 선택하여 결정할것으로 보여짐
            month_type_d = self.find_matrix(train_set_month,train_set_type,prob_c_d,12)
            day_type_d = self.find_matrix(train_set_day,train_set_type,prob_c_d,7)
            time_type_d = self.find_matrix(train_set_time,train_set_type,prob_c_d,6)
            n_type_d = self.find_matrix(train_set_location,train_set_type,prob_c_d,78)
            
            #test_set을 기반으로 각 사건별(월,일,시간,지역) 가장 발생할 확률이 높은 범죄유형을 test_result에 저장한다.
            #그럼 test_result에너느 test_set만큼의 데이터가 범죄유형값으로 들어가게된다.
            test_results = []
            for i in range(len(test_set_month)):
                maxi = 0
                #ex) test_set[0]-> m:5월 d:13일 t:13:00 n:대전시 유성구
                #P(type(1)|m,d,t,n),P(type(2)|m,d,t,n),P(type(3)|m,d,t,n),
                #P(type(4)|m,d,t,n),P(type(5)|m,d,t,n),P(type(6)|m,d,t,n)중
                #가장 확률이 높은 type(범죄유형)을 maxi에 저장하여 
                # test_result에 저장해나가는 식인것같다.
                for j in range(6):
                    k = month_type_d[j][test_set_month[i]-1]*time_type_d[j][test_set_time[i]-1]*day_type_d[j][test_set_day[i]-1]*n_type_d[j][test_set_location[i]-1]
                    if(maxi<k):
                        maxi = k
                        ans = j+1
                test_results.append(ans)
            
            #test_set을 기반으로 각 사건별(월,일,시간,지역)로 가장 발생할 확률이 높은 범죄유형을 test_result에저장
            #test_set_type은 사건별 실제 범죄유형
            return test_results,test_set_type
            #최종적으로 확률을 기반으로 추측한 테스트 데이터의 범죄유형과 테스트케이스의 실제 범죄유형을 비교하여 정확도를 계산한다. 
            #return  calculate_accuracy(test_results,test_c_d)
    
    #data = [month,day,time,location]
    #ex) 12월 11일 3time,Location 4
    #return crime_type
    def simpleBeysianPredict(self,train_set,data):
            train_set_month = train_set['Crime_Month'].values
            train_set_location = train_set['Crime_Location'].values
            train_set_type = train_set['Crime_Type'].values
            train_set_day = train_set['Crime_Day'].values
            train_set_time = train_set['Crime_Time'].values
            
            prob_c_d = self.prob(self.count(train_set_type,6))
            
            month_type_d = self.find_matrix(train_set_month,train_set_type,prob_c_d,12)
            day_type_d = self.find_matrix(train_set_day,train_set_type,prob_c_d,7)
            time_type_d = self.find_matrix(train_set_time,train_set_type,prob_c_d,6)
            n_type_d = self.find_matrix(train_set_location,train_set_type,prob_c_d,78)
            
            maxi = 0
            percent = 0
            for j in range(6):
                k = month_type_d[j][data[0]-1]*time_type_d[j][data[2]-1]*day_type_d[j][data[1]-1]*n_type_d[j][data[3]-1]
                if(maxi<k):
                    maxi = k
                    percent = j+1
                    
            return percent
        
        
    def input_output(self,train_d,test_d):
        x_in = train_d.iloc[:,3:].values
        y_in = train_d.iloc[:,2].values
        y_in = y_in.astype('int')
        x_out = test_d.iloc[:,3:].values
        y_out = test_d.iloc[:,2].values
        y_out = y_out.astype('int')
        return x_in,y_in,x_out,y_out
    
    
    def MultinomialNBTest(self,train_d,test_d):
        x_in,y_in,x_out,y_out = self.input_output(train_d,test_d)
    
        clf = MultinomialNB()
        clf.fit(x_in,y_in)
        t = clf.predict(x_out)

        return t,y_out
    
    def GaussianNBTest(self,train_d,test_d):
    
        x_in,y_in,x_out,y_out = self.input_output(train_d,test_d)
    
        clf = GaussianNB()
        clf.fit(x_in,y_in)
        t = clf.predict(x_out)

        return t,y_out

    def BernoulliNBTest(self,train_d,test_d):

        x_in,y_in,x_out,y_out = self.input_output(train_d,test_d)
    
        clf = BernoulliNB()
        clf.fit(x_in,y_in)
        t = clf.predict(x_out)

        return t,y_out
    
        