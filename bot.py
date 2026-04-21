import os
import json
import html
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, List, Dict, Any, Optional

import requests
import yfinance as yf
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


KST = ZoneInfo("Asia/Seoul")
STATE_PATH = Path("state/alerts.json")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

# 예시:
# WATCHLIST_JSON='[{"name":"Bitcoin","ticker":"BTC-USD"},{"name":"Samsung Electronics","ticker":"005930.KS"}]'
WATCHLIST = json.loads(os.getenv(
    "WATCHLIST_JSON",
    '[{"name":"Bitcoin","ticker":"BTC-USD"},{"name":"Samsung Electronics","ticker":"005930.KS"}]'
))

ALERT_CHANGE_PCT = float(os.getenv("ALERT_CHANGE_PCT", "5.0"))
REALERT_STEP_PCT = float(os.getenv("REALERT_STEP_PCT", "2.0"))

# 항상 최신 preview 모델을 자동 선택하는 것보다,
# 운영 환경에서는 모델명을 고정하는 편이 더 안전합니다.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class PriceSnapshot(BaseModel):
    name: str
    ticker: str
    current_price: float
    previous_close: float
    change_pct: float
    currency_hint: str = ""


class AssetAnalysis(BaseModel):
    ticker: str
    sentiment_score: int = Field(ge=-10, le=10)
    volatility_score: int = Field(ge=0, le=10)
    alert_level: Literal["INFO", "WATCH", "ALERT"]
    summary: str
    positive_factors: List[str]
    negative_factors: List[str]
    risk_factors: List[str]
    short_term_view: str


def now_kst() -> datetime:
    return datetime.now(KST)


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def get_gemini_client() -> genai.Client:
    # 환경변수 GEMINI_API_KEY를 사용합니다.
    return genai.Client()


def choose_flash_model(client: genai.Client) -> str:
    """
    선택적 모델 자동 탐색 함수입니다.
    운영 환경에서는 GEMINI_MODEL 값을 명시적으로 고정하는 것을 추천합니다.
    """
    explicit = os.getenv("GEMINI_MODEL")
    if explicit:
        return explicit

    candidates = []
    for model in client.models.list():
        name = getattr(model, "name", "").replace("models/", "")
        actions = set(getattr(model, "supported_actions", []) or [])

        if (
            "generateContent" in actions
            and "flash" in name.lower()
            and "image" not in name.lower()
            and "tts" not in name.lower()
            and "live" not in name.lower()
        ):
            candidates.append(name)

    preferred_order = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]

    for preferred in preferred_order:
        if preferred in candidates:
            return preferred

    return candidates[0] if candidates else GEMINI_MODEL


def fetch_price(asset: Dict[str, str]) -> Optional[PriceSnapshot]:
    ticker = asset["ticker"]
    name = asset.get("name", ticker)

    try:
        obj = yf.Ticker(ticker)

        daily = obj.history(period="7d", interval="1d", auto_adjust=False)
        intraday = obj.history(period="1d", interval="5m", auto_adjust=False)

        if daily.empty or len(daily["Close"].dropna()) < 2:
            return None

        previous_close = float(daily["Close"].dropna().iloc[-2])

        if not intraday.empty and len(intraday["Close"].dropna()) > 0:
            current_price = float(intraday["Close"].dropna().iloc[-1])
        else:
            current_price = float(daily["Close"].dropna().iloc[-1])

        change_pct = ((current_price - previous_close) / previous_close) * 100

        return PriceSnapshot(
            name=name,
            ticker=ticker,
            current_price=current_price,
            previous_close=previous_close,
            change_pct=change_pct,
        )

    except Exception as e:
        print(f"[WARN] Price fetch failed for {ticker}: {e}")
        return None


def analyze_with_gemini(
    client: genai.Client,
    model_name: str,
    snapshot: PriceSnapshot,
    news_items: List[Dict[str, str]],
) -> AssetAnalysis:
    news_text = "\n".join(
        f"- {item.get('title', '')} / {item.get('source', '')} / {item.get('link', '')}"
        for item in news_items[:10]
    ) or "최근 뉴스가 없습니다."

    prompt = f"""
당신은 AI 투자 리서치 보조 도구입니다.

중요 규칙:
- 직접적인 매수/매도 지시는 하지 마세요.
- 제공된 가격 데이터와 뉴스 데이터만 사용하세요.
- 근거가 약하면 근거가 약하다고 말하세요.
- 가능한 한 한국어로 작성하세요.
- 감성 점수는 -10부터 +10까지입니다.
- 변동성 점수는 0부터 10까지입니다.
- alert_level은 다음 중 하나로 정하세요.
  INFO: 일반적인 상황
  WATCH: 의미 있는 변화가 있지만 긴급하지는 않음
  ALERT: 큰 가격 변동, 심각한 리스크, 또는 시장에 큰 영향을 줄 뉴스

자산:
이름: {snapshot.name}
티커: {snapshot.ticker}
현재가: {snapshot.current_price}
전일 종가: {snapshot.previous_close}
변동률: {snapshot.change_pct:.2f}%

최근 뉴스:
{news_text}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AssetAnalysis,
            temperature=0.2,
            max_output_tokens=1200,
        ),
    )

    return AssetAnalysis.model_validate_json(response.text)


def collect_news_for_asset(asset: Dict[str, str]) -> List[Dict[str, str]]:
    """
    기존에 구현해 둔 Google News RSS 로직을 여기에 연결하면 됩니다.

    반환 형식 예시:
    [
      {"title": "...", "source": "Google News", "link": "..."},
      ...
    ]

    이미 뉴스 수집 로직이 있다고 하셨으므로 여기서는 빈 함수 형태로 남겨둡니다.
    """
    return []


def send_telegram_html(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()


def fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def format_alert(snapshot: PriceSnapshot, analysis: AssetAnalysis) -> str:
    direction = "📈 상승" if snapshot.change_pct >= 0 else "📉 하락"

    return f"""
<b>🚨 투자 모니터링 알림</b>

<b>{html.escape(snapshot.name)} ({html.escape(snapshot.ticker)})</b>
현재가: <code>{fmt_price(snapshot.current_price)}</code>
전일 종가: <code>{fmt_price(snapshot.previous_close)}</code>
변동률: <b>{snapshot.change_pct:+.2f}%</b> ({direction})

<b>AI 판단</b>
감성 점수: <code>{analysis.sentiment_score}/10</code>
변동성 점수: <code>{analysis.volatility_score}/10</code>
알림 단계: <b>{html.escape(analysis.alert_level)}</b>

<b>요약</b>
{html.escape(analysis.summary)}

<b>단기 관점</b>
{html.escape(analysis.short_term_view)}

<b>주요 리스크</b>
{html.escape(" / ".join(analysis.risk_factors[:3]))}

<i>주의: 이 알림은 투자 참고용이며 매수·매도 권유가 아닙니다.</i>
""".strip()


def format_daily_report(items: List[tuple[PriceSnapshot, AssetAnalysis]]) -> str:
    today = now_kst().strftime("%Y-%m-%d %H:%M KST")

    lines = [
        f"<b>📊 AI 투자 브리핑</b>",
        f"<i>{today}</i>",
        "",
    ]

    for snapshot, analysis in items:
        emoji = "🟢" if snapshot.change_pct >= 0 else "🔴"
        lines.extend([
            f"{emoji} <b>{html.escape(snapshot.name)} ({html.escape(snapshot.ticker)})</b>",
            f"현재가: <code>{fmt_price(snapshot.current_price)}</code> / 변동률: <b>{snapshot.change_pct:+.2f}%</b>",
            f"감성: <code>{analysis.sentiment_score}</code> / 변동성: <code>{analysis.volatility_score}</code> / 단계: <b>{html.escape(analysis.alert_level)}</b>",
            f"{html.escape(analysis.summary)}",
            "",
        ])

    lines.append("<i>주의: 이 브리핑은 투자 참고용이며 매수·매도 권유가 아닙니다.</i>")
    return "\n".join(lines)


def should_send_alert(snapshot: PriceSnapshot, state: Dict[str, Any]) -> bool:
    if abs(snapshot.change_pct) < ALERT_CHANGE_PCT:
        return False

    today_key = now_kst().strftime("%Y-%m-%d")
    ticker_state = state.get(snapshot.ticker, {})

    last_alert_date = ticker_state.get("last_alert_date")
    last_alert_abs_change = float(ticker_state.get("last_alert_abs_change_pct", 0))

    # 당일 첫 알림은 발송합니다.
    if last_alert_date != today_key:
        return True

    # 이미 알림을 보낸 후에는 변동폭이 의미 있게 더 커졌을 때만 재알림합니다.
    if abs(snapshot.change_pct) - last_alert_abs_change >= REALERT_STEP_PCT:
        return True

    return False


def mark_alert_sent(snapshot: PriceSnapshot, state: Dict[str, Any]) -> None:
    today_key = now_kst().strftime("%Y-%m-%d")

    state[snapshot.ticker] = {
        "last_alert_date": today_key,
        "last_alert_abs_change_pct": round(abs(snapshot.change_pct), 2),
        "last_price": snapshot.current_price,
        "last_change_pct": round(snapshot.change_pct, 2),
        "updated_at": now_kst().isoformat(),
    }


def run_monitor_mode() -> None:
    client = get_gemini_client()
    model_name = choose_flash_model(client)
    state = load_state()

    for asset in WATCHLIST:
        snapshot = fetch_price(asset)
        if snapshot is None:
            continue

        if not should_send_alert(snapshot, state):
            continue

        news_items = collect_news_for_asset(asset)
        analysis = analyze_with_gemini(client, model_name, snapshot, news_items)

        message = format_alert(snapshot, analysis)
        send_telegram_html(message)

        mark_alert_sent(snapshot, state)
        save_state(state)

        # API 호출이 너무 몰리지 않도록 가볍게 지연합니다.
        time.sleep(1)


def run_daily_report_mode() -> None:
    client = get_gemini_client()
    model_name = choose_flash_model(client)

    report_items = []

    for asset in WATCHLIST:
        snapshot = fetch_price(asset)
        if snapshot is None:
            continue

        news_items = collect_news_for_asset(asset)
        analysis = analyze_with_gemini(client, model_name, snapshot, news_items)
        report_items.append((snapshot, analysis))

        time.sleep(1)

    if report_items:
        send_telegram_html(format_daily_report(report_items))


def main() -> None:
    mode = os.getenv("BOT_MODE", "daily").lower()

    if mode == "monitor":
        run_monitor_mode()
    else:
        run_daily_report_mode()


if __name__ == "__main__":
    main()
