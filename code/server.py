from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_restful import reqparse, abort, Api, Resource
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from twilio.rest import Client
from bs4 import BeautifulSoup
from selenium import webdriver
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import urllib.request
import json
import requests
import os
import sys
import pandas as pd
import chromedriver_autoinstaller
import datetime
import numpy as np

### Function part

subway_code = pd.read_csv('https://storage.googleapis.com/subwaymate/subway_code.csv') # google Drive를 통해 subwaycode 로드

#지하철 정보(string or list or dataFile, json)가 어떤 형식이든 인덱싱을 하면 값을 추출하기 편하다
# data.index와 같이 indexing이 가능한 형식도 있지만 int형이 아니고 모든 데이터형식에 맞는 것도 아니라서 함수 사용
# get_index()함수는 각 데이터를 인덱싱하여 int값으로 리턴해준다.
def get_index():
        counter = 0;
        index = subway_code.index
        for i in index:
            counter += 1
        return counter


# 출발역과 도착역정보의 역코드 필요.
# 따라서 역 이름을 가지고 역 코드를 리턴해주는 함수
def get_station_code(departure, destination):
        start = ''
        stop = ''
        index = get_index()  # get index for json data
        for i in range(index):
                if subway_code['전철역명'][i] == departure and start == '':
                        start = subway_code['외부코드'][i]
                elif subway_code['전철역명'][i] == destination and stop == '':
                        stop = subway_code['외부코드'][i]
        return start, stop


# odsay Lab(대중교통 open API)홈페이지를 이용해서 지하철 정보(소요시간, 경유역, 환승정보, 플랫폼 등)를 가져온다.
# 발급받은 api key를 이용해 URL을 string type으로 만든다.
# 출발역와 도착역을 기준으로 데이터를 불러오기 때문에 그 부분을 비워 놓고 *로 채워 놓는다.(split을 위해서)
def findinfoapi():
    data = request.get_json()
    dep = data['action']['parameters']['departure']['value']
    des = data['action']['parameters']['destination']['value']
    start, stop = get_station_code(dep,des)
    ## 지하철 정보를 받아오기 위해, Odsay Lab api 활용, api 키 인증 필요
    url = "https://api.odsay.com/v1/api/subwayPath?lang=0&CID=1000&SID=*&EID=*&Sopt=1&apiKey=uI%2BTR0JKE161XeAFg0i1qA"
    list_URL = url.split('*')
    URL = list_URL[0] + start + list_URL[1] + stop + list_URL[2]
    response = requests.get(URL)
    text = response.text
    jsonObject = json.loads(text)
    return jsonObject

# 환승역 개수를 counting 하는 함수
def count_transfer():
    obj = findinfoapi()
    if len(obj['result']['driveInfoSet']['driveInfo']) == 1:
        transfer_num = "0"
    else:
        transfer_num = len(obj['result']['exChangeInfoSet']['exChangeInfo'])
    return str(transfer_num)


#출발역과 도착역사이에 몇 개의 역을 지나야 하는지 알려주는 함수
#json형식의 데이터를 받아서 경유역을 모두 리스트에 담아 리턴
def get_stationCount(Object):
    stations = []
    for station in Object['result']['stationSet']['stations']:
        stations.append(station.get('startName'))
    return stations

# 운행정보를 리턴하는 함수
def drive_info():
    obj = findinfoapi()
    departure = obj['result']['globalStartName']
    destination = obj['result']['globalEndName']
    time = obj['result']['globalTravelTime']
    time = str(time)
    global transArr
    transArr = obj['result']['driveInfoSet']['driveInfo']
    if len(transArr) == 1:
        line = []
        for i in range(len(transArr)):
            line.append(transArr[i]["laneName"])
        return line, departure, destination, time
    else:
        transArr1 = obj['result']['exChangeInfoSet']['exChangeInfo']
        line = []
        transfer = []
        for i in range(len(transArr)):
            line.append(transArr[i]["laneName"])
        for j in range(len(transArr1)):
            transfer.append(transArr1[j]['exName'])
        return line, transfer, departure, destination, time


# 환승정보를 리턴하는 함수, 환승횟수에 따라서 각각의 값을 넣어준다, 우리가 찾아본 최대 환승 횟수는 3회이나,
# 만약 환승횟수가 3회를 초과한다면 3회를 기준으로 안내한다.
def transfer_info(transfer_num):
    drive = drive_info()
    if transfer_num == "0":
        line_1 = drive[0][0]
        return line_1, drive[1], drive[2], drive[3],transfer_num
    elif transfer_num == "1":
        line_1 = drive[0][0]
        line_2 = drive[0][1]
        transfer_1 = drive[1][0]
        return line_1, line_2, transfer_1, drive[2], drive[3], drive[4],transfer_num
    elif transfer_num == "2":
        line_1 = drive[0][0]
        line_2 = drive[0][1]
        line_3 = drive[0][2]
        transfer_1 = drive[1][0]
        transfer_2 = drive[1][1]
        return line_1, line_2, line_3, transfer_1, transfer_2, drive[2], drive[3], drive[4],transfer_num
    elif transfer_num == "3":
        line_1 = drive[0][0]
        line_2 = drive[0][1]
        line_3 = drive[0][2]
        line_4 = drive[0][3]
        transfer_1 = drive[1][0]
        transfer_2 = drive[1][1]
        transfer_3 = drive[1][2]
        return line_1, line_2, line_3, line_4, transfer_1, transfer_2, transfer_3, drive[2], drive[3], drive[4],transfer_num
    else:
        line_1 = drive[0][0]
        line_2 = drive[0][1]
        line_3 = drive[0][2]
        line_4 = drive[0][3]
        transfer_1 = drive[1][0]
        transfer_2 = drive[1][1]
        transfer_3 = drive[1][2]
        return line_1, line_2, line_3, line_4, transfer_1, transfer_2, transfer_3, drive[2], drive[3], drive[4],transfer_num


# 날씨 정보를 출력하는 함수
# 네이버 날씨 정보를 웹 크롤링하여 정보를 제공한다.
# 기준은 행당동이며, 서울시가 기준이다. 단, location을 변경하면 동 기준으로 어디든 수정 가능
def weather():
    location = '행당동'
    enc_location = urllib.parse.quote(location + '+날씨')
    url = 'https://search.naver.com/search.naver?ie=utf8&query=' + enc_location
    html = requests.get(url)
    soup = BeautifulSoup(html.text,'html.parser')
    data = soup.find('span',attrs={'class': 'todaytemp'}).text
    whatistheweather = str(data)+'도'
    return whatistheweather


def conges_info(X):
    congestion = pd.read_excel('https://storage.googleapis.com/subwaymate/congestion.xls')
    #혼잡을 결정하는 기준
    congestion_criteria = 0.7
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
            if conges_rate > congestion_criteria:  # stardard to define congestion
                standard.append(1)
            else:
                standard.append(0)
        line_data['기준'] = standard
        return line_data

        #cluster 시각화
    def build_clusters(X,Y):
        conges_degree = pd.DataFrame(columns=('역번호', '혼잡도'))
        conges_degree['역번호'] = X[0].values
        conges_degree['혼잡도'] = Y.values
        k_views = KMeans(n_clusters=3).fit(conges_degree)
        conges_degree['cluster'] = k_views.labels_
        sb.lmplot('역번호','혼잡도', data=conges_degree, fit_reg=False, scatter_kws={"s":100}, hue='cluster')

    def check_week():
        week = datetime.datetime.today().weekday()
        return week

    def get_line(station, lines):
        for i in range(len(lines)):
            j = i+1
            line = str(lines['역명'][i:j].values)
            line = line.replace('[','')
            line = line.replace(']','')
            line = line.replace("'",'')
            if line == station:
                temp = str(lines['호선'][i:j].values)
                return (int(temp[2]) - 1)
        return -1

    congestion = drop_duplicate(congestion)

    if get_line('왕십리',congestion) >= 0:
        line_num = get_line('왕십리',congestion)
    else:
        print("찾는 역이 없습니다.")

    weekdays, saturday, sunday = get_week_weekend(congestion)


    if check_week() < 5:
        week_line = get_weekdays_line(weekdays)
        line_data = week_line[line_num]
    elif check_week() == 5:
        saturday_line = get_weekend_line(saturday)
        line_data = saturday_line[line_num]
    else:
        sunday_line = get_weekend_line(sunday)
        line_data = sunday_line[line_num]

    current_time = get_current_time()

    #머신러닝
    line_data = add_column(current_time, line_data)
    # pyplot
    X, Y, high_congestion = get_X_Y(X, line_data)


    #build_graph(X,Y, line_data)
    #clustering
    XX = pd.DataFrame(line_data['역번호'].values)
    YY = pd.DataFrame(line_data[current_time].values)

    x_train, x_test, y_train, y_test = train_test_split(XX,YY, test_size=0.2,random_state=0)
    k_means = KMeans(n_clusters = 3).fit(x_train)
    cluster_predict = k_means.predict(x_train)
    print(cluster_predict)
    # 선형회귀분석
    clf = LinearRegression()
    clf.fit(x_train, y_train)  #모수 추정
    clf.coef_    #추정된 모수 확인
    clf.intercept_  # 추정 된 상수항 확인
    clf.predict(x_test) # 예측
    clf.score(x_test, y_test) # 모형 성능 평가


    #clustering
    #비지도 학습 모델 Kmeans
    #과거 데이터(역코드와 혼잡도)를 이용해 해당 라인을 3개의 클래스터로 나누고 혼잡도 예측
    #세개의 클래스터로 train데이터에 KMeans 알고리즘 적용
    #train데이터를 시각화하기
    conges_degree = pd.DataFrame(columns=('역번호', '혼잡도'))
    conges_degree['역번호'] = x_train[0].values
    conges_degree['혼잡도'] = y_train.values
    k_means = KMeans(n_clusters=2).fit(conges_degree)
    conges_degree['cluster'] = k_means.labels_
    return high_congestion


# 추가 정보를 처리하는 함수
def addition_info():
    we = weather()
    obj = findinfoapi()
    destination = obj['result']['globalEndName']
    fee_info = obj['result']['cashFare']
		subway_li = get_stationCount(obj) 
		conges = conges_info(subway_li)
		if len(conges) != 0:
        congestion_info = "NUGU 분석결과 현재 시각 이용하시는 노선에 탑승객이 많을 것으로 예상됩니다. 마스크를 철저히 착용해주세요!"
    else:
				congestion_info = ""
    if len(obj['result']['driveInfoSet']['driveInfo']) == 1:
        return destination, str(fee_info), we
    else:
        transArr3 = obj["result"]["exChangeInfoSet"]["exChangeInfo"]
        fast = []
        for i in range(len(transArr3)):
            fast.append(str(transArr3[i]["fastTrain"]) + "다시" + str(transArr3[i]["fastDoor"]))
        var = ','.join(fast)
        var = var.replace('-',"다시")
        fast = var.replace(',',"그리고")
        return fast, destination, str(fee_info), we,congestion_info


# 모든 정보를 모아서, 딕셔너리로 리턴하는 함수, 바디 함수
def content_to_dic():
    global departure
    global destination
    global time, line_1, line_2, line_3
    count = count_transfer()
    show_transfer_info = transfer_info(count)
    show_addition_info = addition_info()
    line_key = ["line_1", "line_2", "line_3", "line_4"]
    transfer_key = ["transfer_1", "transfer_2", "transfer_3","transfer_num"]
    des_dep_key = ["departure", "destination"]
    time_key = ["time"]
    addi_key = ["fast_transfer_info", "destination", "fee_info", "weather_info","congestion_info"]
    arr = []
    if count == "0":
        key0 = [line_key[0], des_dep_key[0], des_dep_key[1], time_key[0],transfer_key[-1]]
        A = dict(zip(key0, show_transfer_info))
        B = dict(zip(addi_key[1:-1], show_addition_info))
        new_dic = {**A, **B}
        return new_dic

    elif count == "1":
        key1 = [line_key[0], line_key[1], transfer_key[0], des_dep_key[0], des_dep_key[1], time_key[0],transfer_key[-1]]
        A = dict(zip(key1, show_transfer_info))
        B = dict(zip(addi_key, show_addition_info))
        new_dic = {**A, **B}
        return new_dic

    elif count == "2":
        key2 = [line_key[0], line_key[1], line_key[2], transfer_key[0], transfer_key[1], des_dep_key[0], des_dep_key[1],
                time_key[0],transfer_key[-1]]
        A = dict(zip(key2, show_transfer_info))
        B = dict(zip(addi_key, show_addition_info))
        new_dic = {**A, **B}
        return new_dic

    else:
        key3 = [line_key[0], line_key[1], line_key[2], line_key[3], transfer_key[0], transfer_key[1], transfer_key[2],
                des_dep_key[0], des_dep_key[1], time_key[0],transfer_key[-1]]
        A = dict(zip(key3, show_transfer_info))
        B = dict(zip(addi_key, show_addition_info))
        new_dic = {**A, **B}
        return new_dic


### Flask Part
#Flask 인스턴스 생성
app = Flask(__name__)
api = Api(app)


#action data

# 각각의 기본 파라미터들을 담은 기본 리스폰스 양식들
def all_param_response():
    response ={
        "version": "2.0",
        "resultCode": "OK",
        "output": {
            "departure": "",
            "destination": "",
            "go_printer": "",
            "transfer_1": "",
            "transfer_2": "",
            "transfer_3": "",
            "line_1": "",
            "line_2": "",
            "line_3": "",
            "line_4": "",
            "time": "",
            "fast_transfer_info": "",
            "congestion_info":"",
            "fee_info": "",
            "weather_info": "",
            "transfer_num": "",
        }
        }
    return response


def get_response():
    response ={
        "version": "2.0",
        "resultCode": "OK",
        "output": {
            "departure": "",
            "destination": "",
            "transfer_num":"",
            "go_printer":"",
            "line_1":"",
            "line_2":"",
            "line_3":"",
            "line_4":"",
            "transfer_1":"",
            "transfer_2":"",
            "transfer_3":"",
            "fast_transfer_info":"",
            "fee_info" : "",
            "weather_info" : "",
            "congestion_info":"",
            "time":""
        }
        }
    return response


def print_dict():
    response ={
        "version": "2.0",
        "resultCode": "OK",
        "output": {
            "departure": "",
            "destination": "",
            "go_printer":"",
            "transfer_num":"",
            "line_1":"",
            "line_2":"",
            "line_3":"",
            "line_4":"",
            "transfer_1":"",
            "transfer_2":"",
            "transfer_3":"",
            "time":"",
            "congestion_info":""
        }
        }
    return response


def addition_dict():
    response ={
        "version": "2.0",
        "resultCode": "OK",
        "output": {
            "departure": "",
            "destination": "",
            "transfer_num":"",
            "line_1":"",
            "line_2":"",
            "line_3":"",
            "line_4":"",
            "transfer_1":"",
            "transfer_2":"",
            "transfer_3":"",
            "fast_transfer_info":"",
            "weather_info":"",
            "fee_info":"",
            "time":"",
            "congestion_info":""
        }
        }
    return response


def addi_dict():
    response ={
        "version": "2.0",
        "resultCode": "OK",
        "output": {
            "departure": "",
            "destination": "",
            "transfer_num":"",
            "line_1":"",
            "line_2":"",
            "line_3":"",
            "line_4":"",
            "transfer_1":"",
            "transfer_2":"",
            "transfer_3":"",
            "time":"",
            "fast_transfer_info":"",
            "fee_info":"",
            "weather_info":"",
            "congestion_info":""
        }
        }
    return response



### 메인 함수
def main_func(trans_num):
    #서버에서 request 받기
    data = request.get_json()
    resour = content_to_dic()
        #response 포맷을 출력해주는 함수 호출
    res = all_param_response()
    key = data['action']['parameters'].keys()
    key = list(key) # 요청온 request에 대해 답해줘야 하는 value의 key값들 (ex. departure, destination, line1,line2, transfer1)
    res_output = res['output'] # departure:"", destination :""
    response = {**res_output,**resour} # 모든 정보가 채워진 response:
    respon_key = response.keys()
    respon_key = list(respon_key)
    data_li = [] # 최종 value 값들의 list
    for i in range(len(key)):
        data_li.append(response[key[i]]) # key의 value 값을 리스트에 담기
    dic1 = dict(zip(key,data_li))
    boolean = 'go_printer' in data['action']['parameters'] # go_printer 여부에 따라 common액션인지 아닌지를 구분
    if(data['action']['actionName']=='addition' or 'notprint'): # 조건문으로 어떤 기본 리스폰스 양식을 불러와야하는지를 선택
        r1 = addition_dict()
    elif(data['action']['actionName']=='print'):
        r1 = print_dict()
    elif(len(data['action']['parameters'])==13):
        r1 = addi_dict()
    r1['output'].update(dic1) #최종으로 response하는 json 값
    if(r1['output']['transfer_num']==''):
        r1['output']['transfer_num'] = trans_num
    if(boolean == True):
        r1['output']['go_printer'] = data['action']['parameters']['go_printer']['value']
    r1 = str(r1) # 리스폰스를 위해 dic -> str(json)으로 형변환
    r1 = r1.replace("'",'"') # restful_api의 기본 규칙을 지키기 위해 ' -> "로 변환
    return r1



##
## URL Router에 맵핑한다.(Rest URL정의)
##

def answer(): # 기본 리스폰스를 위한 함수
    v = count_transfer()
    return main_func(v)

def print_answer(): # 출력에 관련된 리스폰스를 위한 함수, frontend로의 정보전달
    v = count_transfer()
    data = request.get_json()
		jasonObject = findinfoapi()
    dep = data['action']['parameters']['departure']['value']
    des = data['action']['parameters']['destination']['value']
    start, stop = get_station_code(dep,des)
    var1 = main_func(v)
		if data['action']['parameters']['go_printer']['value'] == '출력':
			var2 = render.template('index1.html',image_file="congestion.png",depart=start,desti=stop,data1=jasonObject["result"]['globalTravelTime'],data2=jasonObject["result"]['globalStationCount'],data3=jasonObject['exName'],data4=jasonObject['exWalkTime'],data5=jasonObject['fastDoor'],data6=bistro_name[0],data7=bistro_address[0],data8=bistro_name[1],data9=bistro_address[1],data10=bistro_name[2],data11=bistro_address[2])
		else:
			var2 = ""
    return var2

def healthcheck(): # NUGU Health Check를 위한 함수
    return"200 OK"

# app.add_url_rule를 사용, POST로 통신
app.add_url_rule('/',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_zero',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_one',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_two',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_three',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_zero/ok1',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_one/ok2',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_two/ok3',view_func=answer,methods=['POST'])

app.add_url_rule('/answer1.route/departure_o/transfer_three/ok4',view_func=answer,methods=['POST'])

app.add_url_rule('/addition',view_func=answer,methods=['POST'])

app.add_url_rule('/addition/print',view_func=print_answer,methods=['POST'])

app.add_url_rule('/addition/notprint',view_func=answer,methods=['POST'])

app.add_url_rule('/addition/notprint',view_func=answer,methods=['POST'])

app.add_url_rule('/addition/notprint/a.transfer',view_func=answer,methods=['POST'])

app.add_url_rule('/addition/notprint/a.fee',view_func=answer,methods=['POST'])

app.add_url_rule('/addition/notprint/a.weather',view_func=answer,methods=['POST'])

app.add_url_rule('/health',view_func=healthcheck,methods=['GET'])


### 문자전송을 위한 코드

TWILIO_ACCOUNT_SID = 'AC8def0b9b885cbd7fdd0a34f950567a4c'
TWILIO_AUTH_TOKEN = '223b49fcc6f10bae37bf9881117dbd55'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

client.messages.create(
    #여기에 번호 입력
    to="+8201028045097",
    from_="+18329907655",  #twilio에서 발급받은 번호(don't touch)
    body="우리는 subwayMate 팀입니다"
)


#서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
