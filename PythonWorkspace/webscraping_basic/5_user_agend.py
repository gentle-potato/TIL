import requests
url = "http://nadocoding.tistory.com"
headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
res = requests.get(url, headers=headers)
# res = requests.get(url)
res.raise_for_status()
with open("nadocoding.html", "w", encoding="utf-8") as f:
    f.write(res.text)