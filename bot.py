import google.generativeai as genai
import yfinance as yf
import requests
import os
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime

# 1. 깃허브 금고(Secrets)에서 값 가져오기
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def send_telegram(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ 텔레그램 설정값이 없습니다.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": text})

def get_korean_news(keyword):
    try:
        encoded_query = urllib.parse.quote(keyword + " when:1d") 
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
        response = requests.get(url, timeout=10)
        root = ET.fromstring(response.text)
        titles = [item.find('title').text for item in root.findall('.//item')][:3]
        return "\n".join(titles) if titles else "최신 뉴스가 없습니다."
    except:
        return "뉴스 수집 중 오류가 발생했습니다."

try:
    # 💡 [설정] 여기에 관심 종목을 직접 적으세요 (초보자용)
    interest_items = {
        "비트코인": "BTC-USD",
        "삼성전자": "005930.KS",
        "엔비디아": "NVDA"
    }

    # 💡 [설정] 중요한 D-Day 일정
    d_day_events = {
        "메가터치 첫 출근": "2026-05-04",
        "미국 FOMC 금리발표": "2026-05-01"
    }

    # D-Day 계산
    today = datetime.now().date()
    d_day_msg = "🗓️ [오늘의 주요 일정]\n"
    for ev, dt in d_day_events.items():
        delta = (datetime.strptime(dt, "%Y-%m-%d").date() - today).days
        if delta >= 0: d_day_msg += f"{'🚨 D-Day' if delta==0 else f'👉 D-{delta}'} : {ev}\n"

    # AI 모델 찾기
    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    flash_model = [m for m in models if 'flash' in m.lower() and 'tts' not in m.lower()][0]
    model = genai.GenerativeModel(flash_model.replace("models/", ""))

    report = f"{d_day_msg}\n🔥 [AI 뉴스 & 예측 브리핑]\n"

    for name, ticker in interest_items.items():
        data = yf.Ticker(ticker).history(period="2d")
        price = round(data['Close'].iloc[-1], 2) if not data.empty else "N/A"
        news = get_korean_news(name)
        
        prompt = f"{name}(현재가 {price}) 관련 뉴스: {news}\n위 정보를 보고 1.호재점수(-10~10) 2.변동성예측 3.핵심변수를 딱 3줄로 요약해줘."
        ai_res = model.generate_content(prompt).text
        report += f"\n🏢 {name} ({price})\n{ai_res}\n"

    send_telegram(report)
    print("✅ 모든 분석 전송 완료!")

except Exception as e:
    send_telegram(f"⚠️ 에러 발생: {str(e)}")
    print(f"에러: {e}")
