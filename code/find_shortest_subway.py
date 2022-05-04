#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import datetime
import time

escalator = pd.read_excel('/Users/Eungchan/desktop/Ai_project/escalator.xlsx', header = 2, encoding='utf-8')
escalator_index = escalator.index
distance_time = pd.read_excel('/Users/Eungchan/desktop/Ai_project/distance_time_old.xls',skiprow = 2, encoding = 'utf-8')
distance_index = distance_time.index
transfer_time = pd.read_excel('/Users/Eungchan/desktop/Ai_project/transfer_time.xlsx', encoding = 'utf-8')
transfer_index = transfer_time.index
escalator_speed = 30
#print(escalator)
def get_escalator_time(station):
    check_escalator = 0
    for i in escalator_index:
        if escalator['역명'][i] == station:
            check_escalator = 1
            print("에스컬레이터 올라가는 시간", (round(escalator['승강행정\n(m)'][i]/escalator_speed) * 100), "초")
            break;
    if(check_escalator == 1):
        return(int(round(escalator['승강행정\n(m)'][i]/escalator_speed) * 100))
    else:
        print("에스컬레이터가 없는 역입니다.")
        return 0

def get_distance_time(station, destination):
    check_station = 0
    get_time_distance = 0
    check_direction = 0
    counter = 0
    for i in distance_index:
        counter += 1

    # 정방향일 때 시간 계산
    for i in range(counter):
        if check_station == 0 and distance_time['역명'][i] == station:
            check_station = 1
        if check_station == 1:
            get_time_distance += distance_time['시간\n(분)'][i]
            if distance_time['역명'][i] == destination:
                check_direction = 1
                break
    # 반대방향일 때 소요시간
    if check_direction == 0:
        check_station = 0
        get_time_distance = 0
        for i in range(counter-1, 0, -1):
            if check_station == 0 and distance_time['역명'][i] == station:
                check_station = 1
            if check_station == 1:
                get_time_distance += distance_time['시간\n(분)'][i]
                if distance_time['역명'][i] == destination:
                    check_direction = 1
                    break
    print(station,"-->",destination,int(get_time_distance),"분 소요")
    return int(get_time_distance)
    
def get_transfer_time(station, line):
    counter = 0
    for i in transfer_index:
        counter += 1
    for i in range(counter):
        if transfer_time['환승역명'][i] == station and transfer_time['환승노선'][i] == line:
            timer = transfer_time['환승소요시간(분,초)'][i]
            time_to_list = (str(timer)).split(':')
            print("환승시간", int(time_to_list[1]),"분")
            return int(time_to_list[1])
# main
while True:
    station = input("출발역을 선택하세요 : ")
    if station == "끝":
        break
    destination = input("도착역을 선택하세요 : ")
    print()
    trans_station = "선릉"
    line = '분당선'

    # 2호선 이용하면
    line_time = get_distance_time(station, destination)  # 왕십리 -> 강남
    time.sleep(1)
    escalator_time = get_escalator_time(destination)
    time.sleep(1)
    if escalator_time == 0:
        print("2호선 이용하면 ", line_time,"분")
    elif escalator_time >= 60:
        minutes = escalator_time//60
        seconds = escalator_time % 60
        print("2호선 이용하면 ", line_time + minutes,"분", seconds,"초")
    time.sleep(1)
    print("\n====================\n")
    # 분당선 이용하면
    trans_time = get_distance_time(station, trans_station) # 왕십리 -> 선릉
    time.sleep(1)
    trans_time += get_transfer_time(trans_station, line)  # 환승시간
    time.sleep(1)
    trans_time += get_distance_time(trans_station, destination) # 선릉 -> 강남
    time.sleep(1)
    escalator_time = get_escalator_time(destination)
    time.sleep(1)
    if escalator_time == 0:
        print("분당선 이용하면 ", trans_time,"분")
    elif escalator_time >= 60:
        minutes = escalator_time//60
        seconds = escalator_time % 60
        print("분당선 이용하면 ", trans_time + minutes,"분", seconds,"초")
    print()
    if line_time < trans_time:
        print("2호선이 더 빠릅니다")
    else:
        print("분당선이 더 빠릅니다")



# In[ ]:




