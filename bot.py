import google.generativeai as genai
import yfinance as yf
import requests
import os
import xml.etree.ElementTree as ET
import urllib.parse

# 1. 마스터키 불러오기
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.get(url, params={"chat_id": CHAT_ID, "text": text})

# 💡 업그레이드 무기: 구글 최신 뉴스 실시간 수집기!
def get_korean_news(keyword):
    encoded_query = urllib.parse.quote(keyword + " when:1d") # 최근 24시간 뉴스만
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url)
    root = ET.fromstring(response.text)
    # 최신 뉴스 딱 3개만 핵심으로 뽑아옵니다
    titles = [item.find('title').text for item in root.findall('.//item')][:3]
    return "\n".join(titles) if titles else "최신 뉴스가 없습니다."

try:
    # 안정적인 AI 모델 세팅
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    valid_models = [m for m in available_models if 'flash' in m.lower() and 'tts' not in m.lower() and 'vision' not in m.lower()]
    model = genai.GenerativeModel(valid_models[0].replace("models/", ""))

    # 💡 아이디어 2번 적용: 관심 종목 리스트 (여기에 종목을 계속 추가할 수 있습니다!)
    interest_items = {
        "비트코인": "BTC-USD",
        "삼성전자": "005930.KS"
    }

    final_message = "🔥 [AI 뉴스 & 변동성 예측 브리핑]\n"

    for name, ticker_symbol in interest_items.items():
        # 주가 데이터 가져오기
        hist = yf.Ticker(ticker_symbol).history(period="1d")
        current_price = int(hist['Close'].iloc[-1]) if not hist.empty else 0
        
        # 뉴스 긁어오기
        news_data = get_korean_news(name)
        
        # 💡 아이디어 5번 적용: AI에게 호재/악재 점수 매기기 프롬프트
        prompt = f"""
        너는 주식 시장의 숨은 의도를 파악하는 1% 최고수 트레이더야.
        종목명: {name} (현재가: {current_price})
        
        [최근 24시간 핵심 뉴스]
        {news_data}
        
        위 뉴스를 분석해서 다음 양식에 맞춰 딱 떨어지게 답변해 줘. (주저리주저리 설명 금지)
        
        1. 📊 호재/악재 점수: (강한 악재 -10점 ~ 강한 호재 +10점 중 택 1)
        2. 🎯 주가 변동성 예측: (오늘 주가가 어떻게 튈지 1줄 예측)
        3. 💡 핵심 변수: (투자자가 주의해야 할 리스크나 기회 1줄 요약)
        """
        
        response = model.generate_content(prompt)
        
        # 메시지 누적하기
        final_message += f"\n==================\n"
        final_message += f"🏢 종목: {name}\n"
        final_message += f"💵 현재가: {current_price:,}\n"
        final_message += f"{response.text}\n"

    # 최종 텔레그램 전송
    send_telegram(final_message)
    print("업그레이드 전송 성공!")

except Exception as e:
    send_telegram(f"⚠️ 시스템 에러 발생: {str(e)}")
    print(f"에러 발생: {str(e)}")
