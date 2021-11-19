# *** CH 01 데이터 수집 작업 *** #







##### [1] 파일 수집 #####





##### <2> 파일 데이터 수집 #####


### (1) 키보드 입력 데이터 수집 ###

# ① scan 함수 #
# scan(file, what)
scan("")


### (2) TXT 파일 데이터 수집 ###

# ① read.table 함수 #
# read.table(file, header, sep, fill, what)
df <- read.table(
  file="data.txt",
  header=TRUE
)
df

# ② write.table 함수 #
# write.table(x, file, append, quote, sep, ...)
write.table(iris,
            file="write.txt")
test <- read.table(
  file="write.txt",
  header=TRUE
)
test
remove(test)


### (3) CSV 파일 데이터 수집 ###

# ① read.csv 함수 #
# read.csv(file, header)
csv_data <- read.csv(
  file="data.csv",
  header=TRUE
)
csv_data

# ② write.csv 함수 #
# write.csv(x, file, append, quote, sep, ...)
write.csv(csv_data,
          file="write.csv")
test <- read.csv(
  file="write.csv",
  header=TRUE,
)
test
remove(test)


### (4) 구분자가 포함된 파일 데이터 수집 ###

# ① read.delim 함수 #
# read.delim(file, header)
tsv_data <- read.delim(
  file="data.tsv",
  header=TRUE
)
tsv_data

# ② write.table 함수 #
write.table(
  tsv_data,
  file="write.tsv",
  sep="\t"
)
test <- read.delim(
  file="write.tsv",
  header=TRUE
)
test
remove(test)


### (5) 엑셀 파일 데이터 수집 ###

# ① read_excel 함수 #
# read_excel(path)
# EXCEL에서 데이터 수집 기능을 처음 사용할 경우 readxl 패키지를 설치해야 함
install.packages("readxl")
library(readxl)

excel_data <- read_excel(
  path="data.xlsx",
  sheet="Sheet1",
  range="A1:B4",
  col_names=TRUE
)
View(excel_data)

# ② write.xlsx 함수 #
# write.xlsx(x, file, ...)
# write.xlsx 함수는 openxlsx 패키지를 설치하고 사용해야 함
install.packages("openxlsx")
library(openxlsx)

write.xlsx(
  excel_data,
  file="write.xlsx"
)
test <- read_excel(
  path="write.xlsx",
  sheet="Sheet 1",
  range="A1:B4",
  col_names=TRUE
)
test
View(test)
remove(test)
