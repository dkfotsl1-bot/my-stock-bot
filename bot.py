import google.generativeai as genai
import yfinance as yf
import requests
import os

# 1. 깃허브 금고(Secrets)에서 마스터키 꺼내오기
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": text})

try:
    # 2. 안정적인 AI 모델 찾기
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    valid_models = [m for m in available_models if 'flash' in m.lower() and 'tts' not in m.lower() and 'vision' not in m.lower()]
    best_model_name = valid_models[0].replace("models/", "")
    model = genai.GenerativeModel(best_model_name)

    # 3. 데이터 가져오기 (비트코인, 코스피 지수)
    btc = yf.Ticker("BTC-USD").history(period="5d")['Close'].tolist()
    kospi = yf.Ticker("^KS11").history(period="5d")['Close'].tolist()
    btc_price = int(btc[-1])
    kospi_price = round(kospi[-1], 2)

    # 4. AI 분석 요청
    prompt = f"""
    너는 코스피와 글로벌 가상화폐 시장 흐름을 꿰뚫어보는 최고수 트레이더야.
    최근 5일 비트코인 종가: {btc} (현재 {btc_price}달러)
    최근 5일 코스피 지수 종가: {kospi} (현재 {kospi_price}포인트)
    이 데이터를 바탕으로 오늘 하루 주의해야 할 변수나 포인트를 딱 3줄로 핵심만 요약해줘.
    """
    
    response = model.generate_content(prompt)
    
    # 5. 텔레그램 전송
    msg = f"🌅 [모닝 AI 시장 브리핑]\n\n비트코인: {btc_price:,} 달러\n코스피 지수: {kospi_price} pt\n\n{response.text}"
    send_telegram(msg)
    print("전송 성공!")

except Exception as e:
    send_telegram(f"⚠️ 시스템 에러 발생: {str(e)}")
    print(f"에러 발생: {str(e)}")
