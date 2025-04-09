from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time, os, urllib.parse
from urllib.parse import urlparse

BASE_URL = "https://www.wa.or.kr/"  # 크롤링할 시작 페이지
DATA_DIR = "data"                 # 저장할 폴더
parsed = urlparse(BASE_URL)
domain_name = parsed.netloc.replace('.', '_')  # example.com → example_com

# 디렉토리 생성
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Selenium 브라우저 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 창 없이 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 메인 페이지 접속 및 HTML 가져오기
driver.get(BASE_URL)
time.sleep(2)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# 모든 링크 추출/순서를 유지하면서 중복 제거
seen = set()
ordered_urls = []
for link in soup.find_all('a', href=True):
    url = urllib.parse.urljoin(BASE_URL, link['href'])
    if url.startswith(BASE_URL) and url not in seen:
        seen.add(url)
        ordered_urls.append(url)


print(f"총 {len(ordered_urls)}개의 URL 수집됨")
for i, url in enumerate(ordered_urls):
    try:
        driver.get(url)
        time.sleep(2)

        # 페이지 높이 설정
        scroll_height = driver.execute_script("return document.body.scrollHeight")
        driver.set_window_size(1920, scroll_height)
        time.sleep(1)

        # 파일명 저장
        filename = f"{domain_name}_{i+1:03}.png"
        save_path = os.path.join(DATA_DIR, filename)
        driver.save_screenshot(save_path)
        print(f"[✔] 캡처 완료: {url} → {save_path}")
    except Exception as e:
        print(f"[❌] 실패: {url}\n{e}")

driver.quit()
