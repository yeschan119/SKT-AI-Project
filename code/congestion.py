#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd #데이터프레임 사용
import numpy as np #기초 수학 연산 및 행렬계산
from sklearn import datasets # iris와 같은 내장데이터 사용
from sklearn.model_selection import train_test_split # train, test 데이터 분할
from sklearn.cluster import KMeans #비지도 학습 모델 중 k-menas모듈
from sklearn.linear_model import LinearRegression #선형 회귀분석
import matplotlib.pyplot as plt   # plot 으로 시각화
import seaborn as sb
from sklearn.preprocessing import LabelEncoder   #text를 숫자로 변환해주는 모듈
import datetime
import mglearn
from IPython.core.display import display
#그래프 배경화면을 다크하게
plt.style.use(['dark_background'])
congestion = pd.read_excel('/Users/Eungchan/desktop/Ai_project/2019congestion.xls')
def execute_congestion_check(line_num, X):
    def get_week_weekend(congestion):
        week_index = 0
        saturday_index = 0
        sunday_index = 0
        for day in congestion['조사일자']:
            if day == '평일':
                week_index += 1
            elif day == '토요일':
                saturday_index += 1
            else:
                sunday_index += 1
        saturday_index += week_index
        #평일
        week = congestion.head(week_index)
        #토요일
        saturday = congestion[week_index : saturday_index]
        #일요일
        sunday = congestion.tail(sunday_index)
        return week, saturday, sunday

    def get_weekdays_line(weekdays):
        week_line = []
        temp_line = '1호선'
        index = 0
        start = 0
        for line in weekdays['호선']:
            if line == temp_line:
                index += 1
            else:
                index += 1
                week_line.append(weekdays[start:index-1])
                start = index
                temp_line = line
        week_line.append(weekdays[start:index-1])
        return week_line;

    def drop_duplicate(week):
        for i in range(len(week)):
            if week['구분1'][i] == '상선' or week['구분1'][i] == '외선':
                week.drop(i, inplace = True)
        return week

    def get_weekend_line(weekend):
        weekend_line = []
        temp_line = '1호선'
        index = 0
        start = 0
        for line in weekend['호선']:
            if line == temp_line:
                index += 1
            else:
                index += 1
                weekend_line.append(weekend[start:index-1])
                start = index
                temp_line = line
        weekend_line.append(weekend[start:index-1])
        return weekend_line

    #현재시간을 얻어서 시간과 분만 리턴
    def get_current_time():
        current_time = datetime.datetime.now()
        if int(str(current_time.strftime('%M'))) >= 30:
            current_time = datetime.time(int(str(current_time.strftime('%H'))),30)
        else:
            current_time = datetime.time(int(str(current_time.strftime('%H'))),0)
        return current_time

    def get_X_Y(X, line_data):
        Y = []
        high_congest = []
        for i in range(len(line_data)):
            for station in X:
                if line_data['역명'][i:i+1].values == station:
                    Y.append(line_data[current_time][i:i+1].values)
                    if line_data['기준'][i:i+1].values == 1:
                        high_congest.append(station)
        return X, Y, high_congest

    def add_column(current_time, line_data):
        standard = []
        for conges_rate in line_data[current_time]:
            if conges_rate > 0.35:
                standard.append(1)
            else:
                standard.append(0)
        line_data['기준'] = standard
        return line_data

    def build_graph_with_congestion(X,Y,line_data, high_congestion):
        #그래프 사이즈 조정(pixel단위로)
        fig, ax = plt.subplots()
        x = 700 / fig.dpi
        y = 250 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        #그래프 그리기
        time = datetime.datetime.now().strftime("%H:%M")
        title = str(time)+'분' + " "+line_data['호선'][0:1].values +' 혼잡도'
        new_X = []
        new_Y = []
        index = 0
        for x,y in zip(X,Y):
            markercolor = 'blue'
            color = 'blue'
            new_X.append(x)
            new_Y.append(y)
            index += 1
            if index == 2:
                if new_X[0] in high_congestion and new_X[1] in high_congestion:
                    markercolor = 'red'
                    color = 'red'
                plt.plot(new_X,new_Y,color=color,marker='o',linestyle='solid', markerfacecolor=markercolor,markersize=10)
                del new_X[0]
                del new_Y[0]
                index = 1
    def build_graph(X,Y, line_data):
        fig, ax = plt.subplots()
        x = 700 / fig.dpi
        y = 250 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        #그래프 그리기
        time = datetime.datetime.now().strftime("%H:%M")
        title = str(time)+'분' + " "+line_data['호선'][0:1].values +' 혼잡도'
        plt.plot(X,Y,color='blue',marker='o',linestyle='solid', markerfacecolor='blue',markersize=11)

        plt.title(title,loc='center')
        plt.xlabel('역이름')
        plt.ylabel('혼잡도')
        plt.show()

    #cluster 시각화
    def build_clusters(XX,YY):
        conges_degree = pd.DataFrame(columns=('역번호', '혼잡도'))
        conges_degree['역번호'] = XX[0].values
        conges_degree['혼잡도'] = YY.values
        k_views = KMeans(n_clusters=3).fit(conges_degree)
        conges_degree['cluster'] = k_views.labels_
        sb.lmplot('역번호','혼잡도', data=conges_degree, fit_reg=False, scatter_kws={"s":100}, hue='cluster')


    congestion = drop_duplicate(congestion)
    weekdays, saturday, sunday = get_week_weekend(congestion)
    week_line = get_weekdays_line(weekdays)
    saturday_line = get_weekend_line(saturday)
    sunday_line = get_weekend_line(sunday)
    current_time = get_current_time()

    #머신러닝
    line_data = week_line[line_num]
    line_data = add_column(current_time, line_data)

    # pyplot
    #X = ['왕십리','상왕십리','신당','동대문역사문화공원','을지로4가','을지로3가']
    X, Y, high_congestion = get_X_Y(X, line_data)

    if len(high_congestion) != 0:
        print(high_congestion,"<- 이 역들은 지금 시간에 혼잡합니다.")
    else:
        print("현재시간에는 혼잡한 구간이 없습니다.")

    #한글폰트 삽입 함수
    plt.rc('font', family='AppleGothic')
    #build_graph(X,Y, line_data)
    build_graph_with_congestion(X,Y,line_data,high_congestion)
    #clustering    
    XX = pd.DataFrame(line_data['역번호'].values)
    YY = pd.DataFrame(line_data[current_time].values)

    x_train, x_test, y_train, y_test = train_test_split(XX,YY, test_size=0.2,random_state=0)

    # 선형회귀분석
    clf = LinearRegression()  
    clf.fit(x_train, y_train)  #모수 추정
    clf.coef_    #추정된 모수 확인
    clf.intercept_  # 추정 된 상수항 확인
    clf.predict(x_test) # 예측
    clf.score(x_test, y_test) # 모형 성능 평가
    print("회귀분석 예측결과",clf.score(x_test, y_test))
    #clustering
    #비지도 학습 모델
    k_means = KMeans(n_clusters=2).fit(x_train)
    predict_cluster = k_means.predict(x_test)
    print("비지도학습 예측:",predict_cluster)

    build_clusters(XX,YY)


# In[ ]:




