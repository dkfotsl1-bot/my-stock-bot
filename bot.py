import google.generativeai as genai
import yfinance as yf
import requests
import os
import xml.etree.ElementTree as ET
import urllib.parse
from datetime import datetime

# 1. 마스터키 불러오기
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": text})

def get_korean_news(keyword):
    encoded_query = urllib.parse.quote(keyword + " when:1d") 
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    titles = [item.find('title').text for item in root.findall('.//item')][:3]
    return "\n".join(titles) if titles else "최신 뉴스가 없습니다."

try:
    # 💡 아이디어 3번 적용: 나만의 D-Day 달력 (원하는 일정을 자유롭게 추가하세요!)
    d_day_events = {
        "미국 FOMC 금리 결정": "2026-04-30",
        "메가터치(MEMS MP) 첫 출근": "2026-05-04",
        "관심 공모주(예: HD현대마린) 청약일": "2026-05-08"
    }

    # D-Day 계산기
    today = datetime.now().date()
    d_day_message = "🗓️ [오늘의 주요 D-Day 일정]\n"
    for event, date_str in d_day_events.items():
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        delta = (event_date - today).days
        
        if delta > 0:
            d_day_message += f"👉 D-{delta} : {event}\n"
        elif delta == 0:
            d_day_message += f"🚨 D-Day : {event} (오늘!)\n"
        else:
            pass # 지난 일정은 표시하지 않음

    # AI 모델 세팅
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    valid_models = [m for m in available_models if 'flash' in m.lower() and 'tts' not in m.lower() and 'vision' not in m.lower()]
    model = genai.GenerativeModel(valid_models[0].replace("models/", ""))

    interest_items = {
        "비트코인": "BTC-USD",
        "삼성전자": "005930.KS"
    }

    # 최종 메시지 조립 시작 (D-Day 내용 먼저 상단에 배치)
    final_message = f"{d_day_message}\n🔥 [AI 뉴스 & 변동성 예측 브리핑]\n"

    for name, ticker_symbol in interest_items.items():
        hist = yf.Ticker(ticker_symbol).history(period="1d")
        current_price = int(hist['Close'].iloc[-1]) if not hist.empty else 0
        news_data = get_korean_news(name)
        
        prompt = f"""
        너는 주식 시장의 숨은 의도를 파악하는 1% 최고수 트레이더야.
        종목명: {name} (현재가: {current_price})
        [최근 24시간 핵심 뉴스]
        {news_data}
        위 뉴스를 분석해서 다음 양식에 맞춰 딱 떨어지게 답변해 줘. 
        1. 📊 호재/악재 점수: (-10점 ~ +10점)
        2. 🎯 주가 변동성 예측: (1줄)
        3. 💡 핵심 변수: (1줄)
        """
        
        response = model.generate_content(prompt)
        
        final_message += f"\n==================\n"
        final_message += f"🏢 종목: {name}\n"
        final_message += f"💵 현재가: {current_price:,}\n"
        final_message += f"{response.text}\n"

    # 최종 전송
    send_telegram(final_message)
    print("D-Day 기능 포함 최종 전송 성공!")

except Exception as e:
    send_telegram(f"⚠️ 시스템 에러 발생: {str(e)}")
    print(f"에러 발생: {str(e)}")
