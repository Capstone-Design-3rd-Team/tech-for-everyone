import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import time

# Selenium 설정
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service("C:\\chromedriver-win64\\chromedriver.exe"), options=options)

# 웹페이지 접속
driver.get('https://eclass.seoultech.ac.kr/ilos/main/main_form.acl')
time.sleep(2)

# HTML 저장 및 BeautifulSoup 분석
html = driver.page_source
with open("page.html", "w", encoding="utf-8") as f:
    f.write(html)
soup = BeautifulSoup(html, "html.parser")

# CSS 다운로드
css_links = [link.get_attribute('href') for link in driver.find_elements(By.CSS_SELECTOR, 'link[rel="stylesheet"]')]
css_contents = ""
for i, link in enumerate(css_links):
    try:
        r = requests.get(link, timeout=5)
        r.encoding = r.apparent_encoding
        css_contents += r.text + "\n"
        print(f"✅ CSS {i+1} 다운로드 성공")
    except:
        print(f"⚠️ CSS {i+1} 다운로드 실패: {link}")

# 명암 대비 계산 함수
def get_luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    def channel(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)

def contrast_ratio(rgb1, rgb2):
    lum1 = get_luminance(rgb1)
    lum2 = get_luminance(rgb2)
    L1, L2 = max(lum1, lum2), min(lum1, lum2)
    return (L1 + 0.05) / (L2 + 0.05)

def get_valid_background_color(driver, el):
    while el:
        style = driver.execute_script("""
            const computed = window.getComputedStyle(arguments[0]);
            return {
                backgroundColor: computed.backgroundColor
            };
        """, el)
        bg = style['backgroundColor']
        if bg and not ("rgba(0, 0, 0, 0)" in bg or "transparent" in bg):
            return bg
        el = driver.execute_script("return arguments[0].parentElement;", el)
    return "rgb(255, 255, 255)"  # 기본 배경

# 기준 설정
min_contrast = 4.5
min_font_size_px = 12

# 스타일별 요소들을 묶기 위한 딕셔너리
style_groups = defaultdict(list)

# 전체 요소 가져오기
elements = driver.find_elements(By.XPATH, "//*")

for idx, el in enumerate(elements):
    try:
        text = el.text.strip()
        tag_name = el.tag_name.lower()
        is_button = tag_name == "button"

        has_text = bool(text)
        has_icon = bool(el.find_elements(By.TAG_NAME, 'svg') or el.find_elements(By.TAG_NAME, 'img'))
        has_content = has_text or (is_button and has_icon)

        if not has_content:
            continue  # 텍스트도 아이콘도 없으면 스킵

        # 스타일 추출
        style = driver.execute_script("""
            const computed = window.getComputedStyle(arguments[0]);
            return {
                fontSize: computed.fontSize,
                color: computed.color,
                backgroundColor: computed.backgroundColor
            };
        """, el)

        font_size = style['fontSize']
        color = style['color']
        bg_color = style['backgroundColor']

        # 투명한 배경이면 상위 배경색 추적
        if "rgba" in bg_color and bg_color.endswith(", 0)") or "transparent" in bg_color:
            bg_color = get_valid_background_color(driver, el)

        # 그룹화 키 생성
        key = (font_size, color, bg_color)
        style_groups[key].append((el, idx, text, is_button, has_icon))

    except StaleElementReferenceException:
        print(f"❌ [{idx}] stale element로 인해 건너뜀")
    except Exception as e:
        print(f"❌ [{idx}] 예외 발생: {e}")

markdown_output = ["# 스타일별 명암 대비 및 폰트 분석 결과\n"]

for i, ((font_size, color, bg_color), group) in enumerate(style_groups.items()):
    markdown_output.append(f"## 🎨 스타일 그룹 {i+1}")
    markdown_output.append(f"- **폰트 크기**: `{font_size}`")
    markdown_output.append(f"- **글자색**: `{color}`")
    markdown_output.append(f"- **배경색**: `{bg_color}`")

    rgb_fg = tuple(map(int, re.findall(r'\d+', color)[:3]))
    rgb_bg = tuple(map(int, re.findall(r'\d+', bg_color)[:3]))
    contrast = contrast_ratio(rgb_fg, rgb_bg)
    contrast_text = f"**{contrast:.2f}** (기준: {min_contrast})"

    markdown_output.append(f"- **명암 대비**: {contrast_text}")
    if contrast < min_contrast:
        markdown_output.append(f"  - ⚠️ 명암 대비 부족 (시인성 낮음)")
    else:
        markdown_output.append(f"  - ✅ 명암 대비 충분")

    font_px = float(font_size.replace("px", "").strip())
    if font_px < min_font_size_px:
        markdown_output.append(f"- ⚠️ 글씨 크기 작음 ({font_px}px) → 최소 {min_font_size_px}px 권장")

    markdown_output.append(f"- **요소 목록**:")
    for el, idx, text, is_button, has_icon in group:
        label_info = ""
        if not text and is_button and has_icon:
            label = el.get_attribute("aria-label") or el.get_attribute("title")
            if not label:
                label_info = "⚠️ 대체 텍스트 없음"
            else:
                label_info = f"대체 텍스트: `{label}`"
        display_text = text if text else "(없음)"
        markdown_output.append(f"  - `[{idx}]` 텍스트: **{display_text}** {label_info}")
    markdown_output.append("")  # 줄바꿈

# 마크다운 파일로 저장
with open("style_analysis.md", "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_output))
print("📝 style_analysis.md 파일로 저장 완료 펭! 🐧")
print("\n🎉 스타일별 분석 완료 펭🐧")
