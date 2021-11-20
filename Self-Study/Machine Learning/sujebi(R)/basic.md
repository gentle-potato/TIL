# **R 작업 환경**

- `getwd()` : 현재 위치
- `source("파일명")` : ("파일명"으로) 저장
- `setwd()` : 경로 지정
  - `setwd("C:\Users\khl06\Desktop\Mr.GentleKim\TIL\Self-Study\Machine Learning\sujebi(R)")`
- `q()` : 종료
- `save.image()` : `Ctrl + s`
- `ls()` : 현재 작업공간에 저장된 내용
  - `ls.str()` : Details
- `rm()` : 변수 제거
  - `rm(list=ls())` : 작업공간 전체 변수 제거

- `save(변수명, file="파일명")` : "파일명" 파일에 개별 변수 저장
  - ex) `save(hero, file='hero.rda')`
- `load()` : 저장한 변수 불러오기

