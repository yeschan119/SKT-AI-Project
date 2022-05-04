import requests
import json
import time
from time import gmtime, strftime
import pandas as pd
import xml.etree.ElementTree as ET
import sys
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    from urllib import urlopen
from bs4 import BeautifulSoup

# get realtime_station_arrival API
while True:
    URL = 'http://swopenapi.seoul.go.kr/api/subway/5862464c5079657335396377764572/xml/realtimeStationArrival/ALL/1/5/'
    response = urlopen(URL).read()
    #print(response)
    # parsing the xml
    xtree = ET.fromstring(response)
    rows = []

    # extract field using parsing
    for node in xtree.findall("row"):
        n_subwayId = node.find("subwayId").text
        n_trainLineNm = node.find("trainLineNm").text
        n_subwayHeading = node.find("subwayHeading").text
        n_statnNm = node.find("statnNm").text
        n_recptnDt = node.find("recptnDt").text
        n_arvlMsg2 = node.find("arvlMsg2").text
        rows.append({"역아이디": n_subwayId,
                     "운행노선" : n_trainLineNm,
                     "도착방면": n_subwayHeading,
                     "역이름": n_statnNm,
                     "도착시간": n_recptnDt[10:18],  # only get time, not year, date
                     "도착메시지" : n_arvlMsg2})

    columns = ["역아이디", "운행노선", "도착방면", "역이름","도착시간","도착메시지"]
    real_time_subway = pd.DataFrame(rows, columns = columns)
    print("example : ")
    print(real_time_subway.head(10))
    print()
    print("===============================\n")


    arrival_station = input("역을 입력하세요 : ")
    index = 0
    print()
    get_fild_list = real_time_subway.columns.tolist()
    #for i in get_fild_list:
        #print(i,end="  ")
    print()
    for station_name in real_time_subway['역이름']:
        if station_name == arrival_station:
            df_to_list = real_time_subway.iloc[index].tolist()
            #for i in df_to_list:
                #print(i,end="   ")
            print(real_time_subway.iloc[index])
            time.sleep(1)
            print("\n")
        index += 1
    if index == 0:
        print("찾는 역이 없습니다.")
    rotation = int(input("계속 하시겠습니까?(1/0): "))
    if rotation == 0:
        print('프로그램을 종료합니다.')
        break
