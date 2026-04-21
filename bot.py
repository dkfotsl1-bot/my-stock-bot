import os
import json
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
from google import genai
from google.genai import types


# ============================================================
# 기본 설정
# ============================================================

KST = ZoneInfo("Asia/Seoul")
STATE_PATH = Path("state/alerts.json")

DEFAULT_WATCHLIST_JSON = """
[
  {"name": "Bitcoin", "ticker": "BTC-USD", "query": "Bitcoin BTC"},
  {"name": "Samsung Electronics", "ticker": "005930.KS", "query": "삼성전자"},
  {"name": "SK Hynix", "ticker": "000660.KS", "query": "SK하이닉스"}
]
"""


# ============================================================
# 환경변수 처리
# GitHub Actions에서 비어 있는 Variables가 ""로 들어와도 안전하게 처리합니다.
# ============================================================

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(
            f"필수 환경변수 {name} 값이 비어 있습니다. "
            f"GitHub 저장소의 Settings > Secrets and variables > Actions > Secrets 설정을 확인하세요."
        )
    return value


def get_env_float(name: str, default: float) -> float:
    value = get_env(name, str(default))
    try:
        return float(value)
    except ValueError:
        print(f"[WARN] {name} 값이 숫자가 아닙니다: {value}. 기본값 {default} 사용.")
        return default


GEMINI_API_KEY = require_env("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = require_env("TELEGRAM_CHAT_ID")

# 기본 모델은 수요 폭주가 잦을 수 있는 최신 모델보다 안정성을 우선합니다.
# GitHub Variables에 GEMINI_MODEL을 넣으면 이 값을 덮어쓸 수 있습니다.
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.0-flash")

WATCHLIST = json.loads(get_env("WATCHLIST_JSON", DEFAULT_WATCHLIST_JSON))

ALERT_CHANGE_PCT = get_env_float("ALERT_CHANGE_PCT", 5.0)
REALERT_STEP_PCT = get_env_float("REALERT_STEP_PCT", 2.0)


# ============================================================
# 데이터 모델
# ============================================================

class PriceSnapshot(BaseModel):
    name: str
    ticker: str
    current_price: float
    previous_close: float
    change_pct: float
    currency_hint: str = ""


class AssetAnalysis(BaseModel):
    ticker: str
    sentiment_score: int
    volatility_score: int
    alert_level: Literal["INFO", "WATCH", "ALERT"]
    summary: str
    positive_factors: List[str]
    negative_factors: List[str]
    risk_factors: List[str]
    short_term_view: str


# ============================================================
# 공통 유틸
# ============================================================

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


def clamp_int(value: Any, min_value: int, max_value: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = 0
    return max(min_value, min(max_value, value))


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def get_model_candidates() -> List[str]:
    """
    Gemini가 503 high demand를 반환할 때 다른 Flash 모델로 재시도하기 위한 목록입니다.
    """
    candidates: List[str] = []

    configured = get_env("GEMINI_MODEL")
    if configured:
        candidates.append(configured)

    for model in [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]:
        if model not in candidates:
            candidates.append(model)

    return candidates


def format_price(value: float, ticker: str = "") -> str:
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return f"{value:,.0f}원"
    if value >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def split_long_message(message: str, limit: int = 3500) -> List[str]:
    """
    텔레그램 메시지 길이 제한을 피하기 위해 긴 메시지를 안전하게 나눕니다.
    일반 텍스트만 사용하므로 중간에 잘려도 HTML 오류가 나지 않습니다.
    """
    chunks: List[str] = []
    current = ""

    for line in message.splitlines():
        # 한 줄 자체가 너무 길면 잘라서 넣습니다.
        if len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(line), limit):
                chunks.append(line[i:i + limit])
            continue

        candidate = current + ("\n" if current else "") + line
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# 가격 조회
# ============================================================

def read_fast_info_value(fast_info: Any, keys: List[str]) -> Optional[float]:
    if fast_info is None:
        return None

    for key in keys:
        try:
            if hasattr(fast_info, "get"):
                value = fast_info.get(key)
            else:
                value = getattr(fast_info, key)
            if value is not None:
                return float(value)
        except Exception:
            continue

    return None


def fetch_price(asset: Dict[str, str]) -> Optional[PriceSnapshot]:
    ticker = asset["ticker"]
    name = asset.get("name", ticker)

    try:
        stock = yf.Ticker(ticker)

        fast_info = getattr(stock, "fast_info", None)
        current_price = read_fast_info_value(
            fast_info,
            ["last_price", "lastPrice", "regular_market_price"]
        )
        previous_close = read_fast_info_value(
            fast_info,
            ["previous_close", "previousClose", "regular_market_previous_close"]
        )

        # fast_info가 실패하면 intraday/daily 데이터로 보완합니다.
        intraday = stock.history(period="1d", interval="5m", auto_adjust=False)
        daily = stock.history(period="10d", interval="1d", auto_adjust=False)

        if current_price is None:
            if not intraday.empty and len(intraday["Close"].dropna()) > 0:
                current_price = float(intraday["Close"].dropna().iloc[-1])
            elif not daily.empty and len(daily["Close"].dropna()) > 0:
                current_price = float(daily["Close"].dropna().iloc[-1])

        if previous_close is None:
            closes = daily["Close"].dropna() if not daily.empty else []
            if len(closes) >= 2:
                previous_close = float(closes.iloc[-2])
            elif len(closes) == 1:
                previous_close = float(closes.iloc[-1])

        if current_price is None or previous_close is None or previous_close <= 0:
            print(f"[WARN] 가격 데이터 부족: {name} ({ticker})")
            return None

        change_pct = ((current_price - previous_close) / previous_close) * 100

        return PriceSnapshot(
            name=name,
            ticker=ticker,
            current_price=current_price,
            previous_close=previous_close,
            change_pct=change_pct,
        )

    except Exception as e:
        print(f"[WARN] 가격 조회 실패: {name} ({ticker}) / {e}")
        return None


# ============================================================
# 뉴스 수집: Google News RSS
# ============================================================

def collect_news_for_asset(asset: Dict[str, str], max_items: int = 5) -> List[Dict[str, str]]:
    query = asset.get("query") or asset.get("name") or asset.get("ticker")
    encoded_query = quote_plus(f"{query} when:1d")

    rss_url = (
        "https://news.google.com/rss/search"
        f"?q={encoded_query}"
        "&hl=ko"
        "&gl=KR"
        "&ceid=KR:ko"
    )

    try:
        feed = feedparser.parse(rss_url)
        news_items: List[Dict[str, str]] = []

        for entry in feed.entries[:max_items]:
            news_items.append({
                "title": getattr(entry, "title", ""),
                "source": "Google News",
                "link": getattr(entry, "link", ""),
                "published": getattr(entry, "published", ""),
            })

        return news_items

    except Exception as e:
        print(f"[WARN] 뉴스 수집 실패: {query} / {e}")
        return []


# ============================================================
# Gemini 분석
# ============================================================

def fallback_analysis(snapshot: PriceSnapshot, reason: str = "") -> AssetAnalysis:
    if abs(snapshot.change_pct) >= ALERT_CHANGE_PCT:
        alert_level = "ALERT"
    elif abs(snapshot.change_pct) >= 2:
        alert_level = "WATCH"
    else:
        alert_level = "INFO"

    sentiment = 1 if snapshot.change_pct > 0 else -1 if snapshot.change_pct < 0 else 0
    volatility = clamp_int(round(abs(snapshot.change_pct) * 1.5), 0, 10)

    short_reason = reason[:250] if reason else "Gemini 분석 응답을 처리하지 못했습니다."

    return AssetAnalysis(
        ticker=snapshot.ticker,
        sentiment_score=sentiment,
        volatility_score=volatility,
        alert_level=alert_level,
        summary=(
            f"AI 상세 분석을 생성하지 못해 가격 변동 중심으로 요약했습니다. "
            f"{snapshot.name}의 전일 대비 변동률은 {snapshot.change_pct:+.2f}%입니다."
        ),
        positive_factors=["가격 데이터는 정상적으로 수집되었습니다."],
        negative_factors=[short_reason],
        risk_factors=[
            "뉴스 기반 정성 분석이 제한적입니다.",
            "단기 가격 변동성이 확대될 수 있습니다."
        ],
        short_term_view="추가 뉴스, 거래량, 환율, 지수 흐름을 함께 확인하는 것이 좋습니다."
    )


def normalize_analysis(analysis: AssetAnalysis, ticker: str) -> AssetAnalysis:
    if not analysis.ticker:
        analysis.ticker = ticker

    analysis.sentiment_score = clamp_int(analysis.sentiment_score, -10, 10)
    analysis.volatility_score = clamp_int(analysis.volatility_score, 0, 10)

    if analysis.alert_level not in ["INFO", "WATCH", "ALERT"]:
        analysis.alert_level = "INFO"

    # 너무 긴 문장을 줄여 텔레그램 오류 가능성을 낮춥니다.
    analysis.summary = (analysis.summary or "")[:500]
    analysis.short_term_view = (analysis.short_term_view or "")[:500]
    analysis.positive_factors = [str(x)[:200] for x in (analysis.positive_factors or [])[:3]]
    analysis.negative_factors = [str(x)[:200] for x in (analysis.negative_factors or [])[:3]]
    analysis.risk_factors = [str(x)[:200] for x in (analysis.risk_factors or [])[:3]]

    return analysis


def analyze_with_gemini(
    client: genai.Client,
    snapshot: PriceSnapshot,
    news_items: List[Dict[str, str]],
) -> AssetAnalysis:
    news_text = "\n".join(
        f"- {item.get('title', '')[:180]}"
        for item in news_items[:5]
    ) or "최근 24시간 기준으로 수집된 뉴스가 없습니다."

    prompt = f"""
당신은 개인 투자자를 돕는 AI 투자 리서치 보조 도구입니다.

반드시 지켜야 할 규칙:
- 직접적인 매수, 매도, 보유 지시는 하지 마세요.
- 제공된 가격 데이터와 뉴스 제목만 사용하세요.
- 근거가 약하면 근거가 약하다고 말하세요.
- 한국어로 짧고 명확하게 작성하세요.
- sentiment_score는 -10부터 +10 사이의 정수입니다.
- volatility_score는 0부터 10 사이의 정수입니다.
- alert_level은 INFO, WATCH, ALERT 중 하나입니다.
- 각 리스트는 최대 3개까지만 작성하세요.
- summary와 short_term_view는 각각 2문장 이내로 작성하세요.

분석 대상:
이름: {snapshot.name}
티커: {snapshot.ticker}
현재가: {snapshot.current_price}
전일 종가: {snapshot.previous_close}
전일 대비 변동률: {snapshot.change_pct:+.2f}%

최근 뉴스 제목:
{news_text}
"""

    last_error = ""

    for model_name in get_model_candidates():
        try:
            print(f"[INFO] Gemini 분석 시도: {snapshot.name} / model={model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AssetAnalysis,
                    temperature=0.1,
                    max_output_tokens=2500,
                ),
            )

            parsed = getattr(response, "parsed", None)
            if parsed is not None:
                if isinstance(parsed, AssetAnalysis):
                    return normalize_analysis(parsed, snapshot.ticker)
                return normalize_analysis(AssetAnalysis.model_validate(parsed), snapshot.ticker)

            text = getattr(response, "text", "")
            if not text:
                raise ValueError("Gemini 응답이 비어 있습니다.")

            return normalize_analysis(AssetAnalysis.model_validate_json(text), snapshot.ticker)

        except Exception as e:
            last_error = str(e)
            print(f"[WARN] Gemini 분석 실패: {snapshot.name} ({snapshot.ticker}) / model={model_name} / {e}")
            time.sleep(2)

    return fallback_analysis(snapshot, last_error)


# ============================================================
# 텔레그램 전송
# ============================================================

def send_telegram_message(message: str) -> None:
    """
    안정성을 위해 HTML parse_mode를 사용하지 않습니다.
    이전 400 Bad Request의 가장 흔한 원인이 HTML 파싱 오류이기 때문입니다.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    for chunk in split_long_message(message):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }

        response = requests.post(url, json=payload, timeout=20)

        if not response.ok:
            print("[ERROR] Telegram 전송 실패")
            print(f"[ERROR] status_code={response.status_code}")
            print(f"[ERROR] response={response.text}")
            raise RuntimeError(
                f"Telegram 전송 실패: HTTP {response.status_code}. "
                f"위 response 내용을 확인하세요."
            )

        time.sleep(0.5)


# ============================================================
# 메시지 포맷: 일반 텍스트
# ============================================================

def format_news_lines(news_items: List[Dict[str, str]], max_items: int = 3) -> str:
    if not news_items:
        return "최근 뉴스 없음"

    lines = []

    for idx, item in enumerate(news_items[:max_items], start=1):
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()

        if title:
            lines.append(f"{idx}. {title[:180]}")
        if link:
            lines.append(f"   {link}")

    return "\n".join(lines) if lines else "최근 뉴스 없음"


def format_daily_report(
    items: List[Tuple[PriceSnapshot, AssetAnalysis, List[Dict[str, str]]]]
) -> str:
    today = now_kst().strftime("%Y-%m-%d %H:%M KST")

    lines = [
        "📊 AI 투자 브리핑",
        today,
        "",
        "주의: 이 브리핑은 투자 참고용이며 매수·매도 권유가 아닙니다.",
        "",
        "==============================",
        "",
    ]

    for snapshot, analysis, news_items in items:
        emoji = "🟢" if snapshot.change_pct >= 0 else "🔴"

        lines.extend([
            f"{emoji} {snapshot.name} ({snapshot.ticker})",
            f"현재가: {format_price(snapshot.current_price, snapshot.ticker)}",
            f"전일 종가: {format_price(snapshot.previous_close, snapshot.ticker)}",
            f"변동률: {snapshot.change_pct:+.2f}%",
            f"감성 점수: {analysis.sentiment_score}/10",
            f"변동성 점수: {analysis.volatility_score}/10",
            f"단계: {analysis.alert_level}",
            "",
            "AI 요약",
            analysis.summary,
            "",
            "단기 관점",
            analysis.short_term_view,
            "",
            "주요 리스크",
            " / ".join(analysis.risk_factors[:3]) if analysis.risk_factors else "특이 리스크 없음",
            "",
            "최근 뉴스",
            format_news_lines(news_items),
            "",
            "------------------------------",
            "",
        ])

    return "\n".join(lines)


def format_alert(
    snapshot: PriceSnapshot,
    analysis: AssetAnalysis,
    news_items: List[Dict[str, str]],
) -> str:
    direction = "📈 상승" if snapshot.change_pct >= 0 else "📉 하락"

    return f"""🚨 투자 모니터링 알림

{snapshot.name} ({snapshot.ticker})
현재가: {format_price(snapshot.current_price, snapshot.ticker)}
전일 종가: {format_price(snapshot.previous_close, snapshot.ticker)}
변동률: {snapshot.change_pct:+.2f}% ({direction})

AI 판단
감성 점수: {analysis.sentiment_score}/10
변동성 점수: {analysis.volatility_score}/10
알림 단계: {analysis.alert_level}

요약
{analysis.summary}

단기 관점
{analysis.short_term_view}

주요 리스크
{" / ".join(analysis.risk_factors[:3]) if analysis.risk_factors else "특이 리스크 없음"}

최근 뉴스
{format_news_lines(news_items)}

주의: 이 알림은 투자 참고용이며 매수·매도 권유가 아닙니다.
""".strip()


# ============================================================
# 알림 중복 방지
# ============================================================

def should_send_alert(snapshot: PriceSnapshot, state: Dict[str, Any]) -> bool:
    if abs(snapshot.change_pct) < ALERT_CHANGE_PCT:
        return False

    today_key = now_kst().strftime("%Y-%m-%d")
    ticker_state = state.get(snapshot.ticker, {})

    last_alert_date = ticker_state.get("last_alert_date")
    last_alert_abs_change = float(ticker_state.get("last_alert_abs_change_pct", 0))

    if last_alert_date != today_key:
        return True

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


# ============================================================
# 실행 모드
# ============================================================

def run_daily_report_mode() -> None:
    print("[INFO] daily report mode 시작")
    client = get_gemini_client()

    report_items: List[Tuple[PriceSnapshot, AssetAnalysis, List[Dict[str, str]]]] = []

    for asset in WATCHLIST:
        snapshot = fetch_price(asset)
        if snapshot is None:
            continue

        news_items = collect_news_for_asset(asset)
        analysis = analyze_with_gemini(client, snapshot, news_items)
        report_items.append((snapshot, analysis, news_items))

        time.sleep(1)

    if report_items:
        send_telegram_message(format_daily_report(report_items))
        print("[INFO] daily report 전송 완료")
    else:
        send_telegram_message(
            "⚠️ AI 투자 브리핑\n\n가격 데이터를 가져오지 못해 오늘 리포트를 만들지 못했습니다."
        )
        print("[WARN] report_items 없음")


def run_monitor_mode() -> None:
    print("[INFO] monitor mode 시작")
    client = get_gemini_client()
    state = load_state()
    sent_count = 0

    for asset in WATCHLIST:
        snapshot = fetch_price(asset)
        if snapshot is None:
            continue

        if not should_send_alert(snapshot, state):
            print(f"[INFO] 알림 조건 미충족: {snapshot.name} {snapshot.change_pct:+.2f}%")
            continue

        news_items = collect_news_for_asset(asset)
        analysis = analyze_with_gemini(client, snapshot, news_items)

        send_telegram_message(format_alert(snapshot, analysis, news_items))
        mark_alert_sent(snapshot, state)
        save_state(state)

        sent_count += 1
        time.sleep(1)

    print(f"[INFO] monitor mode 완료 / 발송 알림 수: {sent_count}")


def main() -> None:
    mode = get_env("BOT_MODE", "daily").lower()

    if mode == "monitor":
        run_monitor_mode()
    else:
        run_daily_report_mode()


if __name__ == "__main__":
    main()
