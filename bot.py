import os
import json
import html
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Literal, Tuple
from urllib.parse import quote_plus

import requests
import feedparser
import yfinance as yf
from pydantic import BaseModel
import google.generativeai as genai  # 안정적인 구형 라이브러리 방식으로 전환

# ============================================================
# 기본 설정
# ============================================================
KST = ZoneInfo("Asia/Seoul")
STATE_PATH = Path("state/alerts.json")

# 💡 [중요] 관심 종목을 여기서 직접 수정할 수 있습니다.
DEFAULT_WATCHLIST_JSON = """
[
  {"name": "비트코인", "ticker": "BTC-USD", "query": "비트코인 호재"},
  {"name": "삼성전자", "ticker": "005930.KS", "query": "삼성전자 주가"},
  {"name": "SK하이닉스", "ticker": "000660.KS", "query": "SK하이닉스 반도체"}
]
"""

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default

GEMINI_API_KEY = get_env("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")
WATCHLIST = json.loads(get_env("WATCHLIST_JSON", DEFAULT_WATCHLIST_JSON))

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# 핵심 기능 함수
# ============================================================

def now_kst():
    return datetime.now(KST)

def fetch_price(asset):
    try:
        ticker = asset["ticker"]
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")
        if hist.empty: return None
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2]
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        return {
            "name": asset["name"],
            "ticker": ticker,
            "current": current_price,
            "prev": prev_close,
            "change": change_pct
        }
    except: return None

def get_news(asset):
    query = quote_plus(asset.get("query", asset["name"]) + " when:1d")
    url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        feed = feedparser.parse(url)
        return "\n".join([f"- {e.title}" for e in feed.entries[:3]])
    except: return "뉴스 수집 실패"

def analyze_stock(snapshot, news):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    당신은 전문 투자 분석가입니다. 아래 정보를 보고 3줄 요약 리포트를 작성하세요.
    종목: {snapshot['name']} (변동률: {snapshot['change']:.2f}%)
    최근 뉴스: {news}
    
    양식:
    1. 호재점수: (-10 ~ 10점 사이)
    2. 향후전망: (한 줄 요약)
    3. 주의사항: (한 줄 요약)
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI 분석 일시적 지연"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"전송 실패: {e}")

# ============================================================
# 메인 실행부
# ============================================================

def main():
    print("[INFO] 분석 시작...")
    report = f"<b>📊 AI 투자 브리핑 ({now_kst().strftime('%m/%d %H:%M')})</b>\n"
    
    for asset in WATCHLIST:
        snapshot = fetch_price(asset)
        if not snapshot: continue
        
        news = get_news(asset)
        analysis = analyze_stock(snapshot, news)
        
        emoji = "🔴" if snapshot['change'] < 0 else "🟢"
        report += f"\n{emoji} <b>{snapshot['name']}</b> ({snapshot['change']:.2f}%)\n"
        report += f"현재가: {snapshot['current']:,.0f}\n"
        report += f"{analysis}\n"
        report += "------------------------------\n"
        time.sleep(1)

    send_telegram(report)
    print("[INFO] 완료!")

if __name__ == "__main__":
    main()
