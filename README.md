# SKT-한양-AI project
AI 경진대회

## 목적
  + SKT NUGU 스피커에 지하철 데이터를 이용한 지하철 알림 서비스 탑재
  + 탑승비율 데이터를 활용한 AI 기능 탑재
  + 알림 서비스를 스피커로 알려주는 것과 동시에 문자 서비스 진행
  + 필요한 정보를 정보 취약계층을 위해 페이퍼로 출력해서 제공

## 주요기능
  + 지하철 노선 정보와 소요시간을 알려준다.
  + 사용자의 요청이 있다면, 추가정보(빠른환승 정보, 승객 탑승 비율 정보, 목적지의 날씨 및 주변 맛집 정보) 등을 음성으로 안내한다.
  + 스피커로 알려주었던 정보와 함께 추가정보(빠른환승 정보, 승객 탑승 비율 정보, 목적지의 날씨 및 주변 맛집 정보)를 인쇄하여 제공한다.
  + K-means 알고리즘을 이용하여 서울시 및 서울교통공사에서 제공한 최근 탑승비율 및 혼잡도 데이터를 분석하여 사용자가 이용하는 호선과 목적지까지 탑승비율을 예측하여 제공한다.
  + 가정용 NUGU 캔들을 통해 출발하기전 출발지와 목적지를 Speaker에게 제공하면 문자로 지하철에서 받아볼 수 있는 정보와 동일한 정보를 받아볼 수 있다.

## Members
  + 강응찬, 김진호, 방정하, 임가현, 김건우

## 기여한 부분
  + 필요한 모든 데이터 크롤링 및 정제 with pandas & numpy
  + 데이터 모델링 with K-means

  + Flask를 이용한 서버 구축
## 시연 영상
[![demo-video](<img width="944" alt="스크린샷 2022-05-04 오전 1 25 01" src="https://user-images.githubusercontent.com/83147205/166495770-5a31d76f-d5e0-449b-bc4c-2382e0c597d7.png">)](http://youtu.be/JkwsliFswpM)
## Architecture
![Untitled](https://user-images.githubusercontent.com/83147205/166494222-93752224-acad-4e05-bbaf-8efc2ab3d542.png)


<img width="944" alt="demo" src="https://user-images.githubusercontent.com/83147205/166496686-74336fb0-c4b4-4e6f-b0a9-81abc3ff8afd.png">
