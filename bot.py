
import os
import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional, Literal, Tuple, Set
from urllib.parse import quote_plus, urlparse

import requests
import feedparser
import yfinance as yf
from pydantic import BaseModel
from google import genai
from google.genai import types

KST = ZoneInfo("Asia/Seoul")
STATE_PATH = Path("state/bot_state.json")

DEFAULT_WATCHLIST_JSON = """
[
  {"name": "Bitcoin", "ticker": "BTC-USD", "query": "Bitcoin BTC"},
  {"name": "Samsung Electronics", "ticker": "005930.KS", "query": "삼성전자"},
  {"name": "SK Hynix", "ticker": "000660.KS", "query": "SK하이닉스"}
]
"""


# ============================================================
# 환경변수 처리
# ============================================================

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(f"필수 환경변수 {name} 값이 비어 있습니다.")
    return value


def get_env_float(name: str, default: float) -> float:
    value = get_env(name, str(default))
    try:
        return float(value)
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    value = get_env(name, str(default))
    try:
        return int(value)
    except ValueError:
        return default


def get_env_set(name: str, default_csv: str) -> Set[str]:
    value = get_env(name, default_csv)
    return {x.strip().lower() for x in value.split(",") if x.strip()}


GEMINI_API_KEY = require_env("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = require_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = require_env("TELEGRAM_CHAT_ID")

GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_WATCHLIST = json.loads(get_env("WATCHLIST_JSON", DEFAULT_WATCHLIST_JSON))
ALERT_CHANGE_PCT = get_env_float("ALERT_CHANGE_PCT", 5.0)
REALERT_STEP_PCT = get_env_float("REALERT_STEP_PCT", 2.0)
MAX_NEWS_PER_ASSET = get_env_int("MAX_NEWS_PER_ASSET", 3)
MAX_GEMINI_ASSETS_PER_RUN = get_env_int("MAX_GEMINI_ASSETS_PER_RUN", 1)
ENABLE_GEMINI_MODES = get_env_set("ENABLE_GEMINI_MODES", "premarket,aftermarket")
COMMANDS_ALLOWED_CHAT_IDS = get_env_set("COMMANDS_ALLOWED_CHAT_IDS", TELEGRAM_CHAT_ID)


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
# 공통 유틸 / 상태 저장
# ============================================================

def now_kst() -> datetime:
    return datetime.now(KST)


def default_state() -> Dict[str, Any]:
    return {
        "favorites": DEFAULT_WATCHLIST,
        "last_update_id": 0,
        "alerts": {}
    }


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return default_state()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        merged = default_state()
        merged.update(data or {})
        if not isinstance(merged.get("favorites"), list) or not merged["favorites"]:
            merged["favorites"] = DEFAULT_WATCHLIST
        if not isinstance(merged.get("alerts"), dict):
            merged["alerts"] = {}
        return merged
    except Exception:
        return default_state()


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


def format_price(value: float, ticker: str = "") -> str:
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return f"{value:,.0f}원"
    if value >= 1000:
        return f"{value:,.2f}"
    return f"{value:.4f}"


def split_long_message(message: str, limit: int = 3500) -> List[str]:
    chunks: List[str] = []
    current = ""

    for line in message.splitlines():
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

    return chunks or [""]


def normalize_ticker(raw: str) -> str:
    return raw.strip().upper().replace(" ", "")


def canonical_asset(name: str, ticker: str, query: str) -> Dict[str, str]:
    ticker = normalize_ticker(ticker)
    cleaned_name = (name or ticker).strip()
    cleaned_query = (query or cleaned_name or ticker).strip()
    return {
        "name": cleaned_name,
        "ticker": ticker,
        "query": cleaned_query,
    }


def get_effective_watchlist(state: Dict[str, Any]) -> List[Dict[str, str]]:
    items = state.get("favorites") or DEFAULT_WATCHLIST
    cleaned: List[Dict[str, str]] = []
    seen = set()

    for item in items:
        try:
            asset = canonical_asset(
                item.get("name") or item.get("ticker") or "",
                item.get("ticker") or "",
                item.get("query") or item.get("name") or item.get("ticker") or "",
            )
            if asset["ticker"] not in seen:
                seen.add(asset["ticker"])
                cleaned.append(asset)
        except Exception:
            continue

    return cleaned or DEFAULT_WATCHLIST


def allowed_chat_id(chat_id: str) -> bool:
    return str(chat_id).strip() in COMMANDS_ALLOWED_CHAT_IDS


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


def ticker_exists(asset: Dict[str, str]) -> bool:
    snapshot = fetch_price(asset)
    return snapshot is not None


# ============================================================
# 뉴스 수집
# ============================================================

def extract_source_name(entry: Any) -> str:
    try:
        source = entry.get("source")
        if isinstance(source, dict):
            return str(source.get("title") or "Google News")
        if source and hasattr(source, "title"):
            return str(source.title)
    except Exception:
        pass
    return "Google News"


def resolve_google_news_redirect(url: str) -> str:
    if not url:
        return ""

    if "news.google.com/rss/articles/" not in url:
        return url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10,
            allow_redirects=True,
        )
        final_url = str(response.url).strip()
        final_host = urlparse(final_url).netloc.lower()

        if final_url and "news.google.com" not in final_host:
            return final_url

    except Exception as e:
        print(f"[WARN] 뉴스 링크 정리 실패: {e}")

    return ""


def collect_news_for_asset(
    asset: Dict[str, str],
    max_items: int = 3,
    resolve_links: bool = True,
) -> List[Dict[str, str]]:
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
            raw_link = getattr(entry, "link", "")
            display_link = ""

            if resolve_links:
                display_link = resolve_google_news_redirect(raw_link)

            news_items.append({
                "title": getattr(entry, "title", ""),
                "source": extract_source_name(entry),
                "link": raw_link,
                "display_link": display_link,
                "published": getattr(entry, "published", ""),
            })

        return news_items

    except Exception as e:
        print(f"[WARN] 뉴스 수집 실패: {query} / {e}")
        return []


# ============================================================
# Gemini 분석
# ============================================================

def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY)


def fallback_analysis(snapshot: PriceSnapshot, reason: str = "") -> AssetAnalysis:
    if abs(snapshot.change_pct) >= ALERT_CHANGE_PCT:
        alert_level = "ALERT"
    elif abs(snapshot.change_pct) >= 2:
        alert_level = "WATCH"
    else:
        alert_level = "INFO"

    sentiment = 1 if snapshot.change_pct > 0 else -1 if snapshot.change_pct < 0 else 0
    volatility = clamp_int(round(abs(snapshot.change_pct) * 1.5), 0, 10)

    short_reason = reason[:180] if reason else "AI 분석 대신 가격 변동 중심 요약을 제공합니다."

    return AssetAnalysis(
        ticker=snapshot.ticker,
        sentiment_score=sentiment,
        volatility_score=volatility,
        alert_level=alert_level,
        summary=(
            f"{snapshot.name}의 전일 대비 변동률은 {snapshot.change_pct:+.2f}%입니다. "
            f"AI 상세 분석이 불안정해 가격 기준 요약으로 대체했습니다."
        ),
        positive_factors=["가격 데이터는 정상적으로 수집되었습니다."],
        negative_factors=[short_reason],
        risk_factors=["뉴스 기반 정성 분석이 제한적일 수 있습니다."],
        short_term_view="가격, 거래량, 관련 업종 흐름을 함께 확인하는 것이 좋습니다."
    )


def normalize_analysis(analysis: AssetAnalysis, ticker: str) -> AssetAnalysis:
    if not analysis.ticker:
        analysis.ticker = ticker

    analysis.sentiment_score = clamp_int(analysis.sentiment_score, -10, 10)
    analysis.volatility_score = clamp_int(analysis.volatility_score, 0, 10)

    if analysis.alert_level not in ["INFO", "WATCH", "ALERT"]:
        analysis.alert_level = "INFO"

    analysis.summary = (analysis.summary or "")[:400]
    analysis.short_term_view = (analysis.short_term_view or "")[:300]
    analysis.positive_factors = [str(x)[:140] for x in (analysis.positive_factors or [])[:3]]
    analysis.negative_factors = [str(x)[:140] for x in (analysis.negative_factors or [])[:3]]
    analysis.risk_factors = [str(x)[:140] for x in (analysis.risk_factors or [])[:3]]

    return analysis


def analyze_with_gemini(
    client: genai.Client,
    snapshot: PriceSnapshot,
    news_items: List[Dict[str, str]],
) -> AssetAnalysis:
    news_text = "\n".join(
        f"- {item.get('title', '')[:160]}"
        for item in news_items[:MAX_NEWS_PER_ASSET]
    ) or "최근 24시간 기준으로 수집된 뉴스가 없습니다."

    prompt = f"""
당신은 개인 투자자를 돕는 AI 투자 리서치 보조 도구입니다.

규칙:
- 직접적인 매수/매도 지시는 하지 마세요.
- 제공된 가격과 뉴스 제목만 사용하세요.
- 한국어로 짧고 명확하게 작성하세요.
- sentiment_score는 -10부터 +10 사이 정수
- volatility_score는 0부터 10 사이 정수
- alert_level은 INFO, WATCH, ALERT 중 하나
- 각 리스트는 최대 3개
- summary와 short_term_view는 각각 2문장 이내

자산:
이름: {snapshot.name}
티커: {snapshot.ticker}
현재가: {snapshot.current_price}
전일 종가: {snapshot.previous_close}
전일 대비 변동률: {snapshot.change_pct:+.2f}%

최근 뉴스 제목:
{news_text}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AssetAnalysis,
                temperature=0.1,
                max_output_tokens=1200,
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
        print(f"[WARN] Gemini 분석 실패: {snapshot.name} ({snapshot.ticker}) / {e}")
        return fallback_analysis(snapshot, str(e))


def should_use_gemini(mode: str, index: int) -> bool:
    return (mode.lower() in ENABLE_GEMINI_MODES) and (index < MAX_GEMINI_ASSETS_PER_RUN)


# ============================================================
# Telegram API
# ============================================================

def telegram_api_url(method: str) -> str:
    return f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"


def telegram_post(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(telegram_api_url(method), json=payload, timeout=20)
    data = response.json()

    if not response.ok or not data.get("ok"):
        print(f"[ERROR] Telegram {method} 실패: status={response.status_code} body={response.text}")
        raise RuntimeError(f"Telegram API 실패: {method}")

    return data


def send_telegram_message(
    message: str,
    chat_id: Optional[str] = None,
    reply_markup: Optional[Dict[str, Any]] = None,
    disable_web_page_preview: bool = True,
) -> None:
    target_chat_id = chat_id or TELEGRAM_CHAT_ID

    for chunk in split_long_message(message):
        payload: Dict[str, Any] = {
            "chat_id": target_chat_id,
            "text": chunk,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup

        telegram_post("sendMessage", payload)
        time.sleep(0.4)


def set_bot_commands() -> None:
    commands = [
        {"command": "start", "description": "봇 시작 및 사용법 보기"},
        {"command": "help", "description": "명령어 도움말 보기"},
        {"command": "list", "description": "현재 즐겨찾기 종목 보기"},
        {"command": "add", "description": "종목 추가: /add 티커 이름"},
        {"command": "remove", "description": "종목 제거: /remove 티커"},
        {"command": "report", "description": "즉시 리포트 받기"},
        {"command": "reset", "description": "기본 관심종목으로 되돌리기"},
    ]
    try:
        telegram_post("setMyCommands", {"commands": commands})
    except Exception as e:
        print(f"[WARN] setMyCommands 실패: {e}")


def get_telegram_updates(last_update_id: int) -> List[Dict[str, Any]]:
    payload = {
        "offset": last_update_id + 1,
        "limit": 100,
        "timeout": 0,
        "allowed_updates": ["message"],
    }
    response = telegram_post("getUpdates", payload)
    return response.get("result", [])


def build_news_keyboard(news_items: List[Dict[str, str]], max_items: int = 3) -> Optional[Dict[str, Any]]:
    rows: List[List[Dict[str, str]]] = []

    for idx, item in enumerate(news_items[:max_items], start=1):
        url = (item.get("display_link") or item.get("link") or "").strip()
        if not url:
            continue
        rows.append([
            {
                "text": f"📰 뉴스 {idx} 보기",
                "url": url
            }
        ])

    if not rows:
        return None

    return {"inline_keyboard": rows}


# ============================================================
# 메시지 포맷
# ============================================================

def report_title(report_mode: str) -> str:
    mode = report_mode.lower()
    if mode == "premarket":
        return "🌅 장 시작 전 브리핑"
    if mode == "intraday":
        return "⏱️ 장중 점검 리포트"
    if mode == "aftermarket":
        return "🌙 장마감 브리핑"
    if mode == "monitor":
        return "🚨 실시간 알림"
    return "📊 AI 투자 브리핑"


def format_news_lines(news_items: List[Dict[str, str]], max_items: int = 3) -> str:
    if not news_items:
        return "최근 뉴스 없음"

    lines = []
    for idx, item in enumerate(news_items[:max_items], start=1):
        title = (item.get("title", "") or "").strip()
        source = (item.get("source", "") or "Google News").strip()
        if title:
            lines.append(f"{idx}. {title[:170]}  [{source}]")

    return "\n".join(lines) if lines else "최근 뉴스 없음"


def format_asset_message(
    snapshot: PriceSnapshot,
    analysis: AssetAnalysis,
    news_items: List[Dict[str, str]],
) -> str:
    emoji = "🟢" if snapshot.change_pct >= 0 else "🔴"

    return "\n".join([
        f"{emoji} {snapshot.name} ({snapshot.ticker})",
        f"현재가: {format_price(snapshot.current_price, snapshot.ticker)}",
        f"전일 종가: {format_price(snapshot.previous_close, snapshot.ticker)}",
        f"변동률: {snapshot.change_pct:+.2f}%",
        f"감성 점수: {analysis.sentiment_score}/10",
        f"변동성 점수: {analysis.volatility_score}/10",
        f"단계: {analysis.alert_level}",
        "",
        "요약",
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
    ])


def format_report_header(report_mode: str, watchlist_count: int) -> str:
    return "\n".join([
        report_title(report_mode),
        now_kst().strftime("%Y-%m-%d %H:%M KST"),
        "",
        "주의: 이 브리핑은 투자 참고용이며 매수·매도 권유가 아닙니다.",
        f"대상 종목 수: {watchlist_count}",
        "",
        "뉴스 원문은 각 종목 메시지 아래의 [뉴스 보기] 버튼으로 열 수 있습니다.",
    ])


def format_alert(
    snapshot: PriceSnapshot,
    analysis: AssetAnalysis,
    news_items: List[Dict[str, str]],
) -> str:
    direction = "📈 상승" if snapshot.change_pct >= 0 else "📉 하락"

    return f"""🚨 변동성 알림

{snapshot.name} ({snapshot.ticker})
현재가: {format_price(snapshot.current_price, snapshot.ticker)}
전일 종가: {format_price(snapshot.previous_close, snapshot.ticker)}
변동률: {snapshot.change_pct:+.2f}% ({direction})

요약
{analysis.summary}

최근 뉴스
{format_news_lines(news_items)}

주의: 이 알림은 투자 참고용이며 매수·매도 권유가 아닙니다.
""".strip()


def help_message() -> str:
    return """🤖 사용 가능한 명령어

/add 티커 이름
예) /add NVDA 엔비디아
예) /add 005930.KS 삼성전자
예) /add BTC-USD 비트코인

/remove 티커
예) /remove NVDA

/list
현재 즐겨찾기 종목 보기

/report
즉시 리포트 받기

/reset
기본 관심종목으로 되돌리기

/help
도움말 보기
""".strip()


def format_favorites_list(state: Dict[str, Any]) -> str:
    watchlist = get_effective_watchlist(state)
    lines = ["⭐ 현재 즐겨찾기 종목", ""]
    for idx, asset in enumerate(watchlist, start=1):
        lines.append(f"{idx}. {asset['name']} ({asset['ticker']})")
    return "\n".join(lines)


# ============================================================
# 알림 중복 방지
# ============================================================

def should_send_alert(snapshot: PriceSnapshot, alerts_state: Dict[str, Any]) -> bool:
    asset_threshold = ALERT_CHANGE_PCT

    if abs(snapshot.change_pct) < asset_threshold:
        return False

    today_key = now_kst().strftime("%Y-%m-%d")
    ticker_state = alerts_state.get(snapshot.ticker, {})

    last_alert_date = ticker_state.get("last_alert_date")
    last_alert_abs_change = float(ticker_state.get("last_alert_abs_change_pct", 0))

    if last_alert_date != today_key:
        return True

    if abs(snapshot.change_pct) - last_alert_abs_change >= REALERT_STEP_PCT:
        return True

    return False


def mark_alert_sent(snapshot: PriceSnapshot, alerts_state: Dict[str, Any]) -> None:
    today_key = now_kst().strftime("%Y-%m-%d")

    alerts_state[snapshot.ticker] = {
        "last_alert_date": today_key,
        "last_alert_abs_change_pct": round(abs(snapshot.change_pct), 2),
        "last_price": snapshot.current_price,
        "last_change_pct": round(snapshot.change_pct, 2),
        "updated_at": now_kst().isoformat(),
    }


# ============================================================
# 텔레그램 명령 처리
# ============================================================

def add_favorite(state: Dict[str, Any], ticker: str, name: str = "", query: str = "") -> str:
    asset = canonical_asset(name=name or ticker, ticker=ticker, query=query or name or ticker)

    if not ticker_exists(asset):
        return f"❌ {asset['ticker']} 가격을 찾지 못했습니다. 티커를 다시 확인해 주세요."

    watchlist = get_effective_watchlist(state)
    existing = {item["ticker"] for item in watchlist}
    if asset["ticker"] in existing:
        return f"ℹ️ {asset['ticker']} 는 이미 즐겨찾기에 있습니다."

    watchlist.append(asset)
    state["favorites"] = watchlist
    save_state(state)
    return f"✅ 즐겨찾기에 추가했어요: {asset['name']} ({asset['ticker']})"


def remove_favorite(state: Dict[str, Any], ticker: str) -> str:
    norm = normalize_ticker(ticker)
    watchlist = get_effective_watchlist(state)
    updated = [item for item in watchlist if item["ticker"] != norm]

    if len(updated) == len(watchlist):
        return f"ℹ️ {norm} 는 즐겨찾기 목록에 없어요."

    state["favorites"] = updated or DEFAULT_WATCHLIST
    save_state(state)
    return f"🗑️ 즐겨찾기에서 제거했어요: {norm}"


def reset_favorites(state: Dict[str, Any]) -> str:
    state["favorites"] = DEFAULT_WATCHLIST
    save_state(state)
    return "🔄 즐겨찾기를 기본 관심종목으로 되돌렸어요."


def parse_command(text: str) -> Tuple[str, List[str]]:
    text = (text or "").strip()
    if not text.startswith("/"):
        return "", []

    parts = text.split()
    command = parts[0].split("@")[0].lower()
    args = parts[1:]
    return command, args


def handle_single_message(state: Dict[str, Any], message: Dict[str, Any]) -> None:
    chat = message.get("chat", {})
    chat_id = str(chat.get("id", "")).strip()

    if not chat_id or not allowed_chat_id(chat_id):
        return

    text = (message.get("text") or "").strip()
    command, args = parse_command(text)

    if command in {"/start", "/help"}:
        send_telegram_message(help_message(), chat_id=chat_id)
        send_telegram_message(format_favorites_list(state), chat_id=chat_id)
        return

    if command == "/list":
        send_telegram_message(format_favorites_list(state), chat_id=chat_id)
        return

    if command == "/reset":
        send_telegram_message(reset_favorites(state), chat_id=chat_id)
        send_telegram_message(format_favorites_list(state), chat_id=chat_id)
        return

    if command == "/add":
        if not args:
            send_telegram_message("사용법: /add 티커 이름\n예) /add NVDA 엔비디아", chat_id=chat_id)
            return

        ticker = args[0]
        name = " ".join(args[1:]).strip()
        message_text = add_favorite(state, ticker=ticker, name=name or ticker, query=name or ticker)
        send_telegram_message(message_text, chat_id=chat_id)
        send_telegram_message(format_favorites_list(state), chat_id=chat_id)
        return

    if command == "/remove":
        if not args:
            send_telegram_message("사용법: /remove 티커\n예) /remove NVDA", chat_id=chat_id)
            return

        send_telegram_message(remove_favorite(state, args[0]), chat_id=chat_id)
        send_telegram_message(format_favorites_list(state), chat_id=chat_id)
        return

    if command == "/report":
        run_report_mode("intraday", state=state, target_chat_id=chat_id)
        return

    if text and not text.startswith("/"):
        send_telegram_message(
            "명령어 형식으로 보내주세요.\n예) /add NVDA 엔비디아\n/help 로 전체 명령어 보기",
            chat_id=chat_id
        )
        return


def process_telegram_commands() -> None:
    print("[INFO] commands mode 시작")
    state = load_state()
    set_bot_commands()

    updates = get_telegram_updates(int(state.get("last_update_id", 0)))
    if not updates:
        print("[INFO] 새 텔레그램 명령 없음")
        return

    max_update_id = int(state.get("last_update_id", 0))

    for update in updates:
        update_id = int(update.get("update_id", 0))
        max_update_id = max(max_update_id, update_id)

        message = update.get("message")
        if not message:
            continue

        handle_single_message(state, message)
        time.sleep(0.4)

    state["last_update_id"] = max_update_id
    save_state(state)
    print(f"[INFO] commands mode 완료 / last_update_id={max_update_id}")


# ============================================================
# 실행 모드
# ============================================================

def build_assets_for_report(mode: str, watchlist: List[Dict[str, str]]) -> List[Tuple[PriceSnapshot, List[Dict[str, str]]]]:
    assets_data: List[Tuple[PriceSnapshot, List[Dict[str, str]]]] = []

    for asset in watchlist:
        snapshot = fetch_price(asset)
        if snapshot is None:
            continue

        news_items = collect_news_for_asset(
            asset,
            max_items=MAX_NEWS_PER_ASSET,
            resolve_links=True,
        )

        assets_data.append((snapshot, news_items))
        time.sleep(0.6)

    assets_data.sort(key=lambda x: abs(x[0].change_pct), reverse=True)
    return assets_data


def run_report_mode(mode: str, state: Optional[Dict[str, Any]] = None, target_chat_id: Optional[str] = None) -> None:
    print(f"[INFO] {mode} report mode 시작")

    state = state or load_state()
    watchlist = get_effective_watchlist(state)
    assets_data = build_assets_for_report(mode, watchlist)

    if not assets_data:
        send_telegram_message(
            f"{report_title(mode)}\n\n가격 데이터를 가져오지 못해 리포트를 만들지 못했습니다.",
            chat_id=target_chat_id,
        )
        print("[WARN] assets_data 없음")
        return

    client = None
    if mode.lower() in ENABLE_GEMINI_MODES and MAX_GEMINI_ASSETS_PER_RUN > 0:
        client = get_gemini_client()

    report_items: List[Tuple[PriceSnapshot, AssetAnalysis, List[Dict[str, str]]]] = []

    for idx, (snapshot, news_items) in enumerate(assets_data):
        if client is not None and should_use_gemini(mode, idx):
            analysis = analyze_with_gemini(client, snapshot, news_items)
        else:
            analysis = fallback_analysis(snapshot, "이 시간대 리포트는 빠른 발송을 위해 간단 요약으로 제공합니다.")
        report_items.append((snapshot, analysis, news_items))
        time.sleep(0.3)

    send_telegram_message(
        format_report_header(mode, len(report_items)),
        chat_id=target_chat_id,
    )

    for snapshot, analysis, news_items in report_items:
        send_telegram_message(
            format_asset_message(snapshot, analysis, news_items),
            chat_id=target_chat_id,
            reply_markup=build_news_keyboard(news_items),
            disable_web_page_preview=True,
        )
        time.sleep(0.3)

    print(f"[INFO] {mode} report 전송 완료")


def run_monitor_mode() -> None:
    print("[INFO] monitor mode 시작")
    state = load_state()
    watchlist = get_effective_watchlist(state)
    assets_data = build_assets_for_report("monitor", watchlist)
    alerts_state = state.get("alerts", {})
    sent_count = 0

    for snapshot, news_items in assets_data:
        if not should_send_alert(snapshot, alerts_state):
            print(f"[INFO] 알림 조건 미충족: {snapshot.name} {snapshot.change_pct:+.2f}%")
            continue

        analysis = fallback_analysis(snapshot, "모니터 알림은 속도 우선으로 간단 분석만 사용합니다.")

        send_telegram_message(
            format_alert(snapshot, analysis, news_items),
            reply_markup=build_news_keyboard(news_items),
        )
        mark_alert_sent(snapshot, alerts_state)
        state["alerts"] = alerts_state
        save_state(state)

        sent_count += 1
        time.sleep(0.5)

    print(f"[INFO] monitor mode 완료 / 발송 알림 수: {sent_count}")


def main() -> None:
    mode = get_env("BOT_MODE", "premarket").lower()

    if mode == "monitor":
        run_monitor_mode()
    elif mode == "commands":
        process_telegram_commands()
    elif mode in {"premarket", "intraday", "aftermarket", "daily"}:
        normalized = "premarket" if mode == "daily" else mode
        run_report_mode(normalized)
    else:
        run_report_mode("premarket")


if __name__ == "__main__":
    main()
