#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""
Polymarket Unified Toolkit v1.4.0

Merged features from polymarket-trade v1.0.1 + polymarket-analysis v2.1.0 + smart-money insights + polymarket-odds:
- Market browsing (trending, search, movers)
- Enhanced search via public-search API (v1.3.0)
- CLOB API integration for prices and orderbook (v1.3.0)
- Sports leagues listing (v1.3.0)
- Tags/categories listing (v1.3.0)
- Analysis (arbitrage, momentum)
- Monitoring (price, whale, arb alerts)
- Watchlist with alerts
- Paper trading (buy, sell, portfolio)
- User profile tracking
- Smart Money leaderboard, scoring, signals
- Compact output mode (v1.3.0)

NEW in v1.4.0 - Advanced Academic Analysis:
- Market efficiency analysis (Chen & Pennock 2012)
- Shapley value signal aggregation (Conitzer 2009/2012)
- Combinatorial arbitrage detection (Hanson 2002/2003)

Fixes from v1.0.0 review:
- Fixed undefined variable bugs
- Added proper exception handling
- Added sell command
- Implemented alert_change logic
- Added --json to all commands
- Added API caching and retry
- Unified parameter naming
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import math
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from itertools import permutations, combinations
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# =============================================================================
# Configuration
# =============================================================================

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"  # NEW in v1.3.0

HOME_DIR = Path.home()
DATA_DIR = HOME_DIR / ".polymarket-unified"
CACHE_DIR = DATA_DIR / "cache"
STATE_DIR = DATA_DIR / "state"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"

# Alert thresholds
DEFAULT_PRICE_PCT_THRESHOLD = 5.0      # 5% price change
DEFAULT_WHALE_USD_THRESHOLD = 5000     # $5000 volume spike
DEFAULT_ARB_USD_THRESHOLD = 0.98       # Pair cost < $0.98

# API settings
API_TIMEOUT = 30
API_RETRIES = 3
API_RETRY_DELAY = 1.0
CACHE_TTL_SECONDS = 60


# =============================================================================
# Security Validation Functions
# =============================================================================

def validate_eth_address(addr: str) -> bool:
    """Validate Ethereum address format (0x + 40 hex chars)."""
    if not addr or not isinstance(addr, str):
        return False
    pattern = r'^0x[a-fA-F0-9]{40}$'
    return bool(re.match(pattern, addr.strip()))


def validate_token_id(token_id: str) -> bool:
    """Validate token ID format (alphanumeric with hyphens/underscores, max 100 chars)."""
    if not token_id or not isinstance(token_id, str):
        return False
    if len(token_id) > 100:
        return False
    # Allow alphanumeric, hyphens, underscores
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', token_id))


def validate_slug(slug: str) -> bool:
    """Validate market slug format (alphanumeric with hyphens, max 200 chars)."""
    if not slug or not isinstance(slug, str):
        return False
    if len(slug) > 200:
        return False
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', slug))


def safe_cache_key(key: str) -> str:
    """Generate safe cache key using SHA256 hash to prevent path traversal."""
    return hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]


def safe_price(value, default: float = 0.5) -> float:
    """Safely convert value to price in [0, 1] range."""
    try:
        p = float(value)
        if 0.0 <= p <= 1.0:
            return p
        # Clamp out-of-range values
        return max(0.0, min(1.0, p))
    except (ValueError, TypeError):
        return default


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MarketState:
    """Market state for monitoring."""
    market_id: str
    name: str
    slug: str
    yes_price: float
    no_price: float
    pair_cost: float
    volume: float
    liquidity: float
    end_date: str
    last_check: str


@dataclass
class Alert:
    """Alert from monitoring."""
    type: str
    message: str
    timestamp: str


# =============================================================================
# Storage Utilities
# =============================================================================

def ensure_dirs() -> None:
    """Ensure all data directories exist."""
    for d in [DATA_DIR, CACHE_DIR, STATE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_json(filepath: Path, default: Optional[dict] = None) -> dict:
    """Load JSON file with proper error handling."""
    if filepath.exists():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            print(f"Warning: Failed to load {filepath}: {e}", file=sys.stderr)
    return default if default is not None else {}


def save_json(filepath: Path, data: dict) -> None:
    """Save JSON file with proper error handling."""
    ensure_dirs()
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    except (IOError, OSError) as e:
        print(f"Error: Failed to save {filepath}: {e}", file=sys.stderr)


def load_cache(cache_key: str) -> Optional[dict]:
    """Load cached API response if not expired."""
    safe_key = safe_cache_key(cache_key)
    cache_file = CACHE_DIR / f"{safe_key}.json"
    if not cache_file.exists():
        return None
    
    try:
        data = json.loads(cache_file.read_text())
        cached_time = data.get('_cached_at', 0)
        if time.time() - cached_time > CACHE_TTL_SECONDS:
            return None
        return data.get('payload')
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def save_cache(cache_key: str, payload: Union[dict, list]) -> None:
    """Save API response to cache."""
    ensure_dirs()
    safe_key = safe_cache_key(cache_key)
    cache_file = CACHE_DIR / f"{safe_key}.json"
    try:
        data = {'_cached_at': time.time(), 'payload': payload}
        cache_file.write_text(json.dumps(data, default=str))
    except OSError:
        pass  # Cache failures are non-fatal


# =============================================================================
# API Functions with Retry and Cache
# =============================================================================

def _make_request(url: str, timeout: int = API_TIMEOUT) -> Optional[bytes]:
    """Make HTTP request with retry logic."""
    headers = {"User-Agent": "PolymarketUnified/1.4.0"}
    
    for attempt in range(API_RETRIES):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code == 429:  # Rate limited
                if attempt < API_RETRIES - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                    continue
            print(f"HTTP Error {e.code}: {url}", file=sys.stderr)
            return None
        except URLError as e:
            if attempt < API_RETRIES - 1:
                time.sleep(API_RETRY_DELAY * (attempt + 1))
                continue
            print(f"Network Error: {e}", file=sys.stderr)
            return None
    return None


def fetch_gamma(endpoint: str, params: Optional[dict] = None, use_cache: bool = True) -> Optional[Union[dict, list]]:
    """Fetch from Gamma API with caching and retry."""
    url = f"{GAMMA_API}{endpoint}"
    if params:
        # Validate params keys to prevent injection
        safe_params = {k: str(v) for k, v in params.items() if isinstance(k, str)}
        url += "?" + urlencode(safe_params)
    
    # Use safe cache key to prevent path traversal
    cache_key = safe_cache_key(url)
    if use_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            return cached
    
    # Fetch from API
    data = _make_request(url)
    if data is None:
        return None
    
    try:
        payload = json.loads(data.decode('utf-8'))
        if use_cache:
            save_cache(cache_key, payload)
        return payload
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error: Invalid JSON from {url}: {e}", file=sys.stderr)
        return None


def fetch_data(endpoint: str) -> Optional[Union[dict, list]]:
    """Fetch from Data API with retry."""
    url = f"{DATA_API}{endpoint}"
    
    data = _make_request(url)
    if data is None:
        return None
    
    try:
        return json.loads(data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error: Invalid JSON from {url}: {e}", file=sys.stderr)
        return None


def fetch_clob(endpoint: str) -> Optional[Union[dict, list]]:  # NEW in v1.3.0
    """Fetch from CLOB API with retry."""
    url = f"{CLOB_API}{endpoint}"
    
    data = _make_request(url)
    if data is None:
        return None
    
    try:
        return json.loads(data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error: Invalid JSON from {url}: {e}", file=sys.stderr)
        return None


# =============================================================================
# Formatting Utilities
# =============================================================================

def fmt_price(price: Optional[Union[str, float, int]]) -> str:
    """Format price as percentage."""
    if price is None:
        return "N/A"
    try:
        return f"{float(price) * 100:.1f}%"
    except (ValueError, TypeError):
        return str(price)


def fmt_price_raw(price: Optional[Union[str, float, int]]) -> float:
    """Get raw price as float."""
    if price is None:
        return 0.0
    try:
        return float(price)
    except (ValueError, TypeError):
        return 0.0


def fmt_volume(vol: Optional[Union[str, float, int]]) -> str:
    """Format volume in human readable form."""
    if vol is None:
        return "N/A"
    try:
        v = float(vol)
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        elif v >= 1_000:
            return f"${v/1_000:.1f}K"
        return f"${v:.0f}"
    except (ValueError, TypeError):
        return str(vol)


def fmt_change(change: Optional[Union[str, float, int]]) -> str:
    """Format price change with arrow."""
    if change is None:
        return ""
    try:
        c = float(change) * 100
        if c > 0:
            return f"↑{c:.1f}%"
        elif c < 0:
            return f"↓{abs(c):.1f}%"
        return "→0%"
    except (ValueError, TypeError):
        return ""


def fmt_time_remaining(end_date: Optional[str]) -> str:
    """Format time remaining until end date."""
    if not end_date:
        return ""
    try:
        dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = dt - now
        
        if delta.days < 0:
            return "Ended"
        elif delta.days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                mins = delta.seconds // 60
                return f"{mins}m"
            return f"{hours}h"
        elif delta.days < 7:
            return f"{delta.days}d"
        elif delta.days < 30:
            return f"{delta.days // 7}w"
        return dt.strftime('%b %d')
    except (ValueError, TypeError):
        return ""


# =============================================================================
# Market Resolution
# =============================================================================

def extract_slug(url_or_slug: str) -> str:
    """Extract slug from Polymarket URL or return as-is with validation."""
    if not url_or_slug or not isinstance(url_or_slug, str):
        return ""
    
    if 'polymarket.com' not in url_or_slug:
        # Validate slug format if not a URL
        if validate_slug(url_or_slug):
            return url_or_slug
        return ""
    
    from urllib.parse import urlparse
    parsed = urlparse(url_or_slug)
    path = parsed.path.strip('/')
    
    if path.startswith('event/'):
        path = path[6:]
    
    # Return last segment (market slug)
    if '/' in path:
        slug = path.split('/')[-1]
    else:
        slug = path
    
    # Validate extracted slug
    if validate_slug(slug):
        return slug
    return ""


def resolve_market(url_or_id: str) -> Optional[dict]:
    """Resolve market from URL, slug, or ID with input validation."""
    if not url_or_id or not isinstance(url_or_id, str):
        return None
    
    # Sanitize input
    url_or_id = url_or_id.strip()
    if len(url_or_id) > 200:
        return None
    
    # Try as URL first
    if url_or_id.startswith("http"):
        slug = extract_slug(url_or_id)
        if not slug:
            return None
        data = fetch_gamma("/markets", {"slug": slug, "active": "true"})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
    
    # Try as numeric ID
    if url_or_id.isdigit():
        market = fetch_gamma(f"/markets/{url_or_id}")
        if market and isinstance(market, dict):
            return market
    
    # Try as slug
    if validate_slug(url_or_id):
        data = fetch_gamma("/markets", {"slug": url_or_id, "active": "true"})
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
    
    # Search in active markets (limit scope to prevent abuse)
    data = fetch_gamma("/markets", {"active": "true", "closed": "false", "limit": 100})
    if data and isinstance(data, list):
        search_lower = url_or_id.lower()[:100]  # Limit search scope
        for m in data:
            slug = str(m.get("slug", ""))[:200].lower()
            question = str(m.get("question", ""))[:500].lower()
            if search_lower in slug or search_lower in question:
                return m
    
    return None


def get_prices(market: dict) -> tuple[float, float]:
    """Extract YES and NO prices from market data with validation."""
    prices = market.get('outcomePrices')
    if not prices:
        return 0.5, 0.5  # Default to fair pricing
    
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except json.JSONDecodeError:
            return 0.5, 0.5
    
    if isinstance(prices, list) and len(prices) >= 2:
        try:
            # Use safe_price to ensure valid range
            yes_price = safe_price(prices[0], 0.5)
            no_price = safe_price(prices[1], 0.5)
            return yes_price, no_price
        except (ValueError, TypeError):
            pass
    return 0.5, 0.5


def get_market_id(market: dict) -> str:
    """Get stable market ID."""
    return str(market.get("conditionId") or market.get("id") or "unknown")


# =============================================================================
# Display Formatting
# =============================================================================

def format_market_line(market: dict, compact: bool = False) -> str:  # Updated in v1.3.0
    """Format single market as compact line."""
    question = str(market.get('question', 'Unknown'))[:50]
    yes, no = get_prices(market)
    day_change = fmt_change(market.get('oneDayPriceChange'))
    time_left = fmt_time_remaining(market.get('endDate'))
    vol = fmt_volume(market.get('volume'))
    
    if compact:
        # Compact mode: minimal info
        outcomes = market.get('outcomes')
        if outcomes:
            try:
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                prices = market.get('outcomePrices')
                if isinstance(prices, str):
                    prices = json.loads(prices)
                odds_str = " | ".join([f"{o}: {float(p)*100:.1f}%" for o, p in zip(outcomes, prices)])
                return f"📊 {market.get('question')} | {odds_str} | Vol: {vol}"
            except:
                pass
        return f"📊 {question}: YES {fmt_price(yes)} | Vol: {vol}"
    
    # Standard mode
    parts = [f"• {question}: YES {fmt_price(yes)}"]
    if day_change:
        parts.append(day_change)
    parts.append(f"| Vol: {vol}")
    if time_left:
        parts.append(f"| ⏰ {time_left}")
    
    return " ".join(parts)


def format_market_detail(market: dict, include_slug: bool = True) -> str:
    """Format detailed market view."""
    lines = []
    
    question = market.get('question', 'Unknown')
    lines.append(f"📊 {question}")
    
    yes, no = get_prices(market)
    pair_cost = yes + no
    
    day_change = fmt_change(market.get('oneDayPriceChange'))
    week_change = fmt_change(market.get('oneWeekPriceChange'))
    month_change = fmt_change(market.get('oneMonthPriceChange'))
    
    lines.append(f"   YES: {fmt_price(yes)} {day_change} | NO: {fmt_price(no)}")
    lines.append(f"   1w: {week_change or 'N/A'} | 1m: {month_change or 'N/A'} | Pair: ${pair_cost:.3f}")
    
    bid = market.get('bestBid')
    ask = market.get('bestAsk')
    if bid is not None and ask is not None:
        try:
            spread = float(ask) - float(bid)
            lines.append(f"   Spread: {spread*100:.1f}%")
        except (ValueError, TypeError):
            pass
    
    vol = fmt_volume(market.get('volume'))
    vol_24h = fmt_volume(market.get('volume24hr'))
    lines.append(f"   Volume: {vol} (24h: {vol_24h})")
    
    liq = fmt_volume(market.get('liquidity'))
    if liq:
        lines.append(f"   Liquidity: {liq}")
    
    time_left = fmt_time_remaining(market.get('endDate'))
    if time_left:
        lines.append(f"   ⏰ {time_left}")
    
    if include_slug:
        slug = market.get('slug')
        if slug:
            lines.append(f"   🔗 polymarket.com/event/{slug}")
    
    return '\n'.join(lines)


def format_market_json(market: dict) -> dict:
    """Convert market to JSON-serializable dict."""
    yes, no = get_prices(market)
    return {
        "id": get_market_id(market),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "yes_price": yes,
        "no_price": no,
        "pair_cost": yes + no,
        "volume": market.get("volume"),
        "volume_24h": market.get("volume24hr"),
        "liquidity": market.get("liquidity"),
        "end_date": market.get("endDate"),
        "spread": market.get("bestAsk") and market.get("bestBid") and 
                  float(market.get("bestAsk", 0)) - float(market.get("bestBid", 0)),
        "change_24h": market.get("oneDayPriceChange"),
        "change_1w": market.get("oneWeekPriceChange"),
        "change_1m": market.get("oneMonthPriceChange"),
    }


# =============================================================================
# Core Commands
# =============================================================================

def cmd_trending(args: argparse.Namespace) -> None:
    """Get trending markets by 24h volume."""
    data = fetch_gamma("/events", {
        'order': 'volume24hr',
        'ascending': 'false',
        'closed': 'false',
        'limit': min(args.limit, 100)
    })
    
    if not data or not isinstance(data, list):
        print("Failed to fetch trending markets", file=sys.stderr)
        return
    
    if args.json:
        output = []
        for event in data[:args.limit]:
            markets = event.get('markets', [])
            output.append({
                "title": event.get("title"),
                "slug": event.get("slug"),
                "volume": event.get("volume"),
                "volume_24h": event.get("volume24hr"),
                "end_date": event.get("endDate"),
                "markets": [format_market_json(m) for m in markets[:3]]
            })
        print(json.dumps(output, indent=2))
        return
    
    compact = getattr(args, 'compact', False)
    
    if compact:
        print(f"🔥 Trending Markets (Top {min(len(data), args.limit)})\n")
        for event in data[:args.limit]:
            markets = event.get('markets', [])
            for m in markets[:2]:
                print(format_market_line(m, compact=True))
        return
    
    print(f"🔥 Trending Markets (Top {min(len(data), args.limit)})\n")
    
    for event in data[:args.limit]:
        title = event.get('title', 'Unknown')
        vol = fmt_volume(event.get('volume'))
        vol_24h = fmt_volume(event.get('volume24hr'))
        time_left = fmt_time_remaining(event.get('endDate'))
        
        print(f"• {title}")
        print(f"  Vol: {vol} (24h: {vol_24h}) | ⏰ {time_left}")
        
        markets = event.get('markets', [])
        for m in markets[:3]:
            print(f"    {format_market_line(m)}")
        if len(markets) > 3:
            print(f"    ... and {len(markets)-3} more markets")
        print()


def cmd_search(args: argparse.Namespace) -> None:
    """Search markets - uses public-search API if available, fallback to local filter."""
    query = args.query.lower().strip()
    
    if not query:
        print("Error: Empty search query", file=sys.stderr)
        return
    
    # NEW in v1.3.0: Try public-search endpoint first (from polymarket-odds)
    public_search = fetch_gamma("/public-search", {"q": query, "limit": 50}, use_cache=False)
    if public_search and isinstance(public_search, dict) and public_search.get('events'):
        events = [e for e in public_search['events'] if e.get('active') and not e.get('closed')]
        
        if events:
            if args.json:
                output = []
                for event in events[:args.limit]:
                    markets = event.get('markets', [])
                    output.append({
                        "title": event.get("title"),
                        "slug": event.get("slug"),
                        "volume": event.get("volume"),
                        "markets": [format_market_json(m) for m in markets[:3]]
                    })
                print(json.dumps(output, indent=2))
                return
            
            compact = getattr(args, 'compact', False)
            if compact:
                print(f"🔍 Search: '{args.query}' ({len(events)} matches)\n")
                for event in events[:args.limit]:
                    markets = event.get('markets', [])
                    for m in markets[:2]:
                        print(format_market_line(m, compact=True))
                return
            
            print(f"🔍 Search: '{args.query}' ({len(events)} matches)\n")
            for event in events[:args.limit]:
                print(format_event_from_search(event))
            return
    
    # Fallback to original search method
    slug = query.replace(' ', '-')
    data = fetch_gamma("/markets", {"slug": slug, "active": "true", "limit": 20})
    
    if data and isinstance(data, list) and len(data) > 0:
        if args.json:
            print(json.dumps([format_market_json(m) for m in data[:args.limit]], indent=2))
            return
        
        compact = getattr(args, 'compact', False)
        if compact:
            for m in data[:args.limit]:
                print(format_market_line(m, compact=True))
            return
        
        print(f"🔍 Search: '{args.query}'\n")
        for m in data[:args.limit]:
            print(format_market_detail(m))
            print()
        return
    
    # Search broader
    data = fetch_gamma("/markets", {"active": "true", "closed": "false", "limit": 200}, use_cache=True)
    if not data or not isinstance(data, list):
        print("Failed to fetch markets", file=sys.stderr)
        return
    
    matches = []
    for m in data:
        text = f"{m.get('question','')} {m.get('slug','')} {m.get('description','')}".lower()
        if query in text:
            matches.append(m)
    
    if not matches:
        print(f"No markets found for '{args.query}'")
        return
    
    if args.json:
        print(json.dumps([format_market_json(m) for m in matches[:args.limit]], indent=2))
        return
    
    compact = getattr(args, 'compact', False)
    if compact:
        print(f"🔍 Search: '{args.query}' ({len(matches)} matches)\n")
        for m in matches[:args.limit]:
            print(format_market_line(m, compact=True))
        return
    
    print(f"🔍 Search: '{args.query}' ({len(matches)} matches)\n")
    for m in matches[:args.limit]:
        print(format_market_detail(m))
        print()


def format_event_from_search(event: dict) -> str:
    """Format event from public-search response."""
    lines = []
    
    title = event.get('title', 'Unknown')
    lines.append(f"\n🎯 {title}")
    
    vol = fmt_volume(event.get('volume'))
    lines.append(f"   Volume: {vol}")
    
    markets = event.get('markets', [])
    for m in markets[:5]:
        if m.get('active') and not m.get('closed'):
            lines.append(format_market_line(m))
    
    if len(markets) > 5:
        lines.append(f"   ... and {len(markets) - 5} more markets")
    
    return '\n'.join(lines)


def cmd_movers(args: argparse.Namespace) -> None:
    """Show biggest price movers."""
    tf_map = {'24h': 'oneDayPriceChange', '1w': 'oneWeekPriceChange', '1m': 'oneMonthPriceChange'}
    change_field = tf_map.get(args.timeframe, 'oneDayPriceChange')
    
    data = fetch_gamma("/markets", {
        "active": "true",
        "closed": "false",
        "limit": 200
    }, use_cache=True)
    
    if not data or not isinstance(data, list):
        print("Failed to fetch markets", file=sys.stderr)
        return
    
    # Calculate absolute change, filter by volume
    min_vol = args.min_volume * 1000  # Convert to USD
    markets_with_change = []
    
    for m in data:
        change = m.get(change_field)
        if change is None:
            continue
        try:
            abs_change = abs(float(change))
            vol = float(m.get('volume') or 0)
            if vol >= min_vol and abs_change > 0.001:  # At least 0.1% change
                markets_with_change.append((m, abs_change, float(change)))
        except (ValueError, TypeError):
            continue
    
    markets_with_change.sort(key=lambda x: x[1], reverse=True)
    
    if args.json:
        output = []
        for m, _, raw in markets_with_change[:args.limit]:
            j = format_market_json(m)
            j["change_selected"] = raw
            output.append(j)
        print(json.dumps(output, indent=2))
        return
    
    compact = getattr(args, 'compact', False)
    if compact:
        print(f"📈 Movers ({args.timeframe})\n")
        for m, _, raw_chg in markets_with_change[:args.limit]:
            print(format_market_line(m, compact=True))
        return
    
    print(f"📈 Biggest Movers ({args.timeframe}, min ${args.min_volume}K volume)\n")
    
    for m, _, raw_chg in markets_with_change[:args.limit]:
        direction = "↑" if raw_chg > 0 else "↓"
        print(f"{direction} {format_market_line(m)}")


def cmd_event(args: argparse.Namespace) -> None:
    """Get specific event/market details."""
    market = resolve_market(args.market)
    if not market:
        print(f"Error: Market not found: {args.market}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(format_market_json(market), indent=2))
        return
    
    print(format_market_detail(market))


# =============================================================================
# NEW in v1.3.0 - Commands from polymarket-odds
# =============================================================================

def cmd_tags(args: argparse.Namespace) -> None:
    """List available market categories/tags."""
    data = fetch_gamma("/tags", {"limit": args.limit})
    
    if data is None or not isinstance(data, list):
        print("Failed to fetch tags", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(data, indent=2))
        return
    
    print("Available Categories\n")
    print(f"{'Name':<25} {'Slug':<20} {'Count':<10}")
    print("-" * 55)
    
    for tag in data:
        label = tag.get('label', 'Unknown')
        slug = tag.get('slug', '')
        count = tag.get('marketCount', tag.get('count', 'N/A'))
        print(f"{label:<25} {slug:<20} {count}")
    
    print(f"\n💡 Use with: events --tag=<slug>")


def cmd_sports(args: argparse.Namespace) -> None:
    """List available sports leagues/series."""
    data = fetch_gamma("/sports")
    
    if data is None or not isinstance(data, list):
        print("Sports endpoint not available or empty", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(data, indent=2))
        return
    
    print("Sports Leagues\n")
    print(f"{'Name':<30} {'Series ID':<15}")
    print("-" * 45)
    
    for sport in data[:args.limit]:
        title = sport.get('title', sport.get('label', 'Unknown'))
        series_id = sport.get('id', '')
        print(f"{title:<30} {series_id}")
    
    print(f"\n💡 Use with: events --series=<series_id>")


def cmd_price(args: argparse.Namespace) -> None:
    """Get current price from CLOB for a token with validation."""
    token_id = args.token_id
    if not validate_token_id(token_id):
        print(f"Error: Invalid token ID format", file=sys.stderr)
        sys.exit(1)
    
    side = args.side
    data = fetch_clob(f"/price?token_id={token_id}&side={side}")
    
    if data is None:
        print(f"Failed to fetch price for token {token_id}", file=sys.stderr)
        sys.exit(1)
    
    price = float(data.get('price', 0))
    
    if args.json:
        print(json.dumps({"token_id": token_id, "side": side, "price": price, "price_pct": price * 100}, indent=2))
        return
    
    print(f"Price ({side}): {price * 100:.1f}%")


def cmd_book(args: argparse.Namespace) -> None:
    """Get orderbook depth from CLOB for a token with validation."""
    token_id = args.token_id
    if not validate_token_id(token_id):
        print(f"Error: Invalid token ID format", file=sys.stderr)
        sys.exit(1)
    
    data = fetch_clob(f"/book?token_id={token_id}")
    
    if data is None:
        print(f"Failed to fetch orderbook for token {token_id}", file=sys.stderr)
        sys.exit(1)
    
    if args.json:
        print(json.dumps(data, indent=2))
        return
    
    print(f"Orderbook for token {token_id}\n")
    
    bids = data.get('bids', [])
    asks = data.get('asks', [])
    
    print("Bids:")
    if bids:
        for b in bids[:5]:
            p = float(b.get('price', 0)) * 100
            s = b.get('size', 0)
            print(f"  {p:.1f}% x ${s}")
    else:
        print("  None")
    
    print("\nAsks:")
    if asks:
        for a in asks[:5]:
            p = float(a.get('price', 0)) * 100
            s = a.get('size', 0)
            print(f"  {p:.1f}% x ${s}")
    else:
        print("  None")


# =============================================================================
# Analysis Commands
# =============================================================================

def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze market for trading edges."""
    market = resolve_market(args.market)
    if not market:
        print(f"Error: Market not found: {args.market}", file=sys.stderr)
        sys.exit(1)
    
    yes, no = get_prices(market)
    pair_cost = yes + no
    
    result = {
        "market": format_market_json(market),
        "analysis": {}
    }
    
    # Arbitrage check
    arb_threshold = args.arb_threshold
    if pair_cost < arb_threshold:
        profit = (1 - pair_cost) * 100
        result["analysis"]["arbitrage"] = {
            "opportunity": True,
            "pair_cost": pair_cost,
            "potential_profit_pct": profit
        }
    else:
        result["analysis"]["arbitrage"] = {
            "opportunity": False,
            "pair_cost": pair_cost
        }
    
    # Momentum
    result["analysis"]["momentum"] = {
        "24h": market.get("oneDayPriceChange"),
        "1w": market.get("oneWeekPriceChange"),
        "1m": market.get("oneMonthPriceChange")
    }
    
    # Liquidity assessment
    liq = market.get("liquidity") or 0
    vol = market.get("volume") or 0
    result["analysis"]["liquidity"] = {
        "tier": "high" if liq > 100000 else "medium" if liq > 10000 else "low",
        "liquidity": liq,
        "volume": vol
    }
    
    if args.json:
        print(json.dumps(result, indent=2))
        return
    
    # Human-readable output
    print(f"📊 Analysis: {market.get('question', 'Unknown')}\n")
    
    print("Market Data:")
    print(f"  YES: {fmt_price(yes)} | NO: {fmt_price(no)}")
    print(f"  Volume: {fmt_volume(market.get('volume'))} (24h: {fmt_volume(market.get('volume24hr'))})")
    print(f"  Liquidity: {fmt_volume(market.get('liquidity'))}")
    print()
    
    print("Arbitrage Check:")
    if result["analysis"]["arbitrage"]["opportunity"]:
        print(f"  ✅ OPPORTUNITY: Pair cost ${pair_cost:.4f} < ${arb_threshold}")
        print(f"  Potential profit: {profit:.2f}%")
    else:
        print(f"  No arb: Pair cost ${pair_cost:.4f} (threshold: ${arb_threshold})")
    print()
    
    print("Momentum:")
    mom = result["analysis"]["momentum"]
    print(f"  24h: {fmt_change(mom['24h']) or 'N/A'}")
    print(f"  1w:  {fmt_change(mom['1w']) or 'N/A'}")
    print(f"  1m:  {fmt_change(mom['1m']) or 'N/A'}")


def cmd_monitor(args: argparse.Namespace) -> None:
    """Monitor market for alerts."""
    market = resolve_market(args.market)
    if not market:
        print(f"Error: Market not found: {args.market}", file=sys.stderr)
        sys.exit(1)
    
    market_id = get_market_id(market)
    yes, no = get_prices(market)
    
    current = MarketState(
        market_id=market_id,
        name=market.get("question", "Unknown"),
        slug=market.get("slug", ""),
        yes_price=yes,
        no_price=no,
        pair_cost=yes + no,
        volume=float(market.get("volume") or 0),
        liquidity=float(market.get("liquidity") or 0),
        end_date=market.get("endDate", ""),
        last_check=datetime.now(timezone.utc).isoformat()
    )
    
    # Load previous state
    state_file = STATE_DIR / f"{market_id}.json"
    prev_data = load_json(state_file)
    
    if prev_data:
        previous = MarketState(**prev_data)
    else:
        previous = None
    
    # Detect alerts
    alerts: list[Alert] = []
    price_threshold = args.price_threshold / 100  # Convert % to decimal
    whale_threshold = args.whale_threshold
    arb_threshold = args.arb_threshold
    
    if previous:
        # Price change alert
        if previous.yes_price > 0:
            change = abs(current.yes_price - previous.yes_price) / previous.yes_price
            if change >= price_threshold:
                direction = "up" if current.yes_price > previous.yes_price else "down"
                alerts.append(Alert(
                    type="PRICE",
                    message=f"YES moved {direction} {change*100:.1f}%: ${previous.yes_price:.3f} → ${current.yes_price:.3f}",
                    timestamp=current.last_check
                ))
        
        # Whale alert (volume spike)
        vol_delta = current.volume - previous.volume
        if vol_delta >= whale_threshold:
            alerts.append(Alert(
                type="WHALE",
                message=f"Large volume: +${vol_delta:,.0f} (total: ${current.volume:,.0f})",
                timestamp=current.last_check
            ))
    
    # Arbitrage alert (stateless - always check)
    if current.pair_cost < arb_threshold:
        profit = (1 - current.pair_cost) * 100
        alerts.append(Alert(
            type="ARBITRAGE",
            message=f"Pair cost ${current.pair_cost:.4f} → {profit:.2f}% profit opportunity",
            timestamp=current.last_check
        ))
    
    # Save state
    save_json(state_file, asdict(current))
    
    # Output
    result = {
        "market": asdict(current),
        "alerts": [asdict(a) for a in alerts],
        "previous": asdict(previous) if previous else None
    }
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"## {current.name}")
        print(f"YES: ${current.yes_price:.3f} | NO: ${current.no_price:.3f} | Pair: ${current.pair_cost:.3f}")
        print(f"Volume: ${current.volume:,.0f}")
        
        if alerts:
            print("\n🚨 Alerts")
            for a in alerts:
                print(f"  [{a.type}] {a.message}")
        else:
            print("\nNo alerts triggered.")
    
    # Exit with alert count for cron
    sys.exit(len(alerts))


def cmd_profile(args: argparse.Namespace) -> None:
    """Fetch user profile with wallet address validation."""
    wallet = args.wallet.lower().strip()
    if not validate_eth_address(wallet):
        print("Error: Invalid Ethereum wallet address", file=sys.stderr)
        sys.exit(1)
    
    positions = fetch_data(f"/positions?user={wallet}")
    
    if positions is None:
        print(f"Failed to fetch profile for {wallet[:10]}...{wallet[-6:]}", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(positions, list):
        positions = []
    
    # Build result
    result = {
        "wallet": wallet,
        "positions": [],
        "summary": {"total_pnl": 0.0, "position_count": len(positions)}
    }
    
    for p in positions:
        pos_data = {
            "market": p.get("title", p.get("market", "Unknown")),
            "outcome": p.get("outcome", "?"),
            "size": float(p.get("size", 0)),
            "avg_price": float(p.get("avgPrice", 0)),
            "current_price": float(p.get("curPrice", p.get("currentPrice", 0))),
            "pnl": 0.0
        }
        pos_data["pnl"] = (pos_data["current_price"] - pos_data["avg_price"]) * pos_data["size"]
        result["summary"]["total_pnl"] += pos_data["pnl"]
        result["positions"].append(pos_data)
    
    if args.json:
        print(json.dumps(result, indent=2))
        return
    
    # Human-readable
    print(f"# Profile: {wallet[:10]}...{wallet[-6:]}\n")
    
    if not result["positions"]:
        print("No open positions.")
        return
    
    print("| Market | Side | Shares | Entry | Current | P&L |")
    print("|--------|------|--------|-------|---------|-----|")
    
    for p in result["positions"]:
        market_name = p["market"][:35] if len(p["market"]) > 35 else p["market"]
        print(f"| {market_name} | {p['outcome']} | {p['size']:.1f} | "
              f"${p['avg_price']:.3f} | ${p['current_price']:.3f} | ${p['pnl']:+.2f} |")
    
    total = result["summary"]["total_pnl"]
    print(f"\n**Total P&L:** ${total:+.2f}")


# =============================================================================
# Watchlist Commands
# =============================================================================

def cmd_watch(args: argparse.Namespace) -> None:
    """Manage watchlist."""
    watchlist = load_json(WATCHLIST_FILE, {"markets": []})
    
    if args.action == "list":
        if not watchlist.get("markets"):
            print("Watchlist is empty")
            return
        
        if args.json:
            # Enrich with current prices
            output = []
            for item in watchlist["markets"]:
                market = resolve_market(item.get("slug", ""))
                if market:
                    yes, _ = get_prices(market)
                    entry = {
                        "slug": item["slug"],
                        "current_price_pct": yes * 100,
                        "alert_at_pct": item.get("alert_at"),
                        "alert_change_pct": item.get("alert_change"),
                        "added": item.get("added")
                    }
                    output.append(entry)
            print(json.dumps(output, indent=2))
            return
        
        print("Watchlist\n")
        for item in watchlist["markets"]:
            market = resolve_market(item.get("slug", ""))
            alert_parts = []
            
            if item.get("alert_at"):
                alert_parts.append(f"alert@{item['alert_at']}%")
            if item.get("alert_change"):
                alert_parts.append(f"±{item['alert_change']}%")
            
            alert_str = f" [{', '.join(alert_parts)}]" if alert_parts else ""
            
            if market:
                yes, _ = get_prices(market)
                print(f"• {market.get('question', item['slug'])}: {fmt_price(yes)}{alert_str}")
            else:
                print(f"• {item['slug']}: (not found){alert_str}")
    
    elif args.action == "add":
        if not args.market:
            print("Error: Market slug/URL required for add", file=sys.stderr)
            sys.exit(1)
        
        market = resolve_market(args.market)
        if not market:
            print(f"Error: Market not found: {args.market}", file=sys.stderr)
            sys.exit(1)
        
        slug = market.get("slug") or args.market
        
        # Check for duplicates
        for item in watchlist.get("markets", []):
            if item.get("slug") == slug:
                print(f"Already watching: {slug}")
                return
        
        entry = {
            "slug": slug,
            "added": datetime.now().isoformat()
        }
        if args.alert_at is not None:
            entry["alert_at"] = args.alert_at
        if args.alert_change is not None:
            entry["alert_change"] = args.alert_change
        
        watchlist["markets"].append(entry)
        save_json(WATCHLIST_FILE, watchlist)
        print(f"✅ Added to watchlist: {market.get('question', slug)}")
    
    elif args.action == "remove":
        if not args.market:
            print("Error: Market slug required for remove", file=sys.stderr)
            sys.exit(1)
        
        slug = extract_slug(args.market)
        original_count = len(watchlist.get("markets", []))
        watchlist["markets"] = [m for m in watchlist.get("markets", []) if m.get("slug") != slug]
        
        if len(watchlist["markets"]) < original_count:
            save_json(WATCHLIST_FILE, watchlist)
            print(f"✅ Removed from watchlist: {slug}")
        else:
            print(f"Not found in watchlist: {slug}")


def cmd_alerts(args: argparse.Namespace) -> None:
    """Check watchlist for alerts."""
    watchlist = load_json(WATCHLIST_FILE, {"markets": []})
    
    if not watchlist.get("markets"):
        if not args.quiet:
            print("Watchlist is empty")
        return
    
    triggered = []
    
    for item in watchlist["markets"]:
        market = resolve_market(item.get("slug", ""))
        if not market:
            continue
        
        yes, _ = get_prices(market)
        current_price = yes * 100  # as percentage
        
        # Check price target alert
        if item.get("alert_at") is not None:
            target = float(item["alert_at"])
            # Use 2% tolerance or 5% of target, whichever is larger
            tolerance = max(2.0, target * 0.05)
            if abs(current_price - target) <= tolerance:
                triggered.append({
                    "market": market.get("question", item["slug"]),
                    "slug": item["slug"],
                    "type": "TARGET",
                    "current_price": current_price,
                    "target_price": target,
                    "message": f"Price {current_price:.1f}% near target {target}%"
                })
        
        # Check price change alert (requires history)
        # Load previous price from state if available
        market_id = get_market_id(market)
        state_file = STATE_DIR / f"{market_id}.json"
        state = load_json(state_file)
        
        if state and item.get("alert_change") is not None:
            prev_price = state.get("yes_price", 0) * 100
            if prev_price > 0:
                change_threshold = float(item["alert_change"])
                price_change = abs(current_price - prev_price)
                if price_change >= change_threshold:
                    direction = "up" if current_price > prev_price else "down"
                    triggered.append({
                        "market": market.get("question", item["slug"]),
                        "slug": item["slug"],
                        "type": "CHANGE",
                        "current_price": current_price,
                        "previous_price": prev_price,
                        "change": price_change,
                        "message": f"Price moved {direction} {price_change:.1f}%: {prev_price:.1f}% → {current_price:.1f}%"
                    })
        
        # Update state for next comparison
        if state is None:
            state = {}
        state["yes_price"] = yes
        state["last_check"] = datetime.now().isoformat()
        save_json(state_file, state)
    
    if args.json:
        print(json.dumps(triggered, indent=2))
        return
    
    if triggered:
        print("🚨 Watchlist Alerts\n")
        for t in triggered:
            print(f"[{t['type']}] {t['market']}: {t['message']}")
    elif not args.quiet:
        print("No alerts triggered.")


# =============================================================================
# Paper Trading Commands
# =============================================================================

def cmd_buy(args: argparse.Namespace) -> None:
    """Paper trade: buy position."""
    market = resolve_market(args.market)
    if not market:
        print(f"Error: Market not found: {args.market}", file=sys.stderr)
        sys.exit(1)
    
    yes, no = get_prices(market)
    price = yes if args.outcome == "yes" else no
    
    if price <= 0:
        print("Error: Invalid market price", file=sys.stderr)
        sys.exit(1)
    
    portfolio = load_json(PORTFOLIO_FILE, {"cash": 10000.0, "positions": []})
    
    cost = float(args.amount)
    if cost <= 0:
        print("Error: Amount must be positive", file=sys.stderr)
        sys.exit(1)
    
    if cost > portfolio["cash"]:
        print(f"Error: Insufficient cash: ${portfolio['cash']:.2f} available, ${cost:.2f} needed", file=sys.stderr)
        sys.exit(1)
    
    shares = cost / price
    
    portfolio["cash"] -= cost
    portfolio["positions"].append({
        "market_slug": market.get("slug", args.market),
        "market_question": market.get("question", "Unknown"),
        "outcome": args.outcome,
        "shares": shares,
        "entry_price": price,
        "cost": cost,
        "date": datetime.now().isoformat(),
        "sold": False
    })
    
    save_json(PORTFOLIO_FILE, portfolio)
    print(f"✅ Bought ${cost:.2f} of {args.outcome.upper()} @ {fmt_price(price)}")
    print(f"   Shares: {shares:.4f} | Cash remaining: ${portfolio['cash']:.2f}")


def cmd_sell(args: argparse.Namespace) -> None:
    """Paper trade: sell position."""
    portfolio = load_json(PORTFOLIO_FILE, {"cash": 10000.0, "positions": []})
    
    if not portfolio.get("positions"):
        print("No positions to sell")
        return
    
    # Find positions for this market that aren't sold
    target_slug = extract_slug(args.market)
    matching_positions = []
    other_positions = []
    
    for pos in portfolio["positions"]:
        if pos.get("market_slug") == target_slug and not pos.get("sold"):
            matching_positions.append(pos)
        else:
            other_positions.append(pos)
    
    if not matching_positions:
        print(f"No open positions found for {args.market}")
        return
    
    # Get current price
    market = resolve_market(args.market)
    if not market:
        print(f"Warning: Market not found, using entry price for {args.market}")
        current_price = matching_positions[0]["entry_price"]
    else:
        yes, no = get_prices(market)
        current_price = yes if matching_positions[0]["outcome"] == "yes" else no
    
    # Calculate total sale
    total_shares = sum(p["shares"] for p in matching_positions)
    total_cost = sum(p["cost"] for p in matching_positions)
    sale_value = total_shares * current_price
    pnl = sale_value - total_cost
    
    # Mark as sold
    for pos in matching_positions:
        pos["sold"] = True
        pos["exit_price"] = current_price
        pos["exit_date"] = datetime.now().isoformat()
        pos["pnl"] = pos["shares"] * current_price - pos["cost"]
    
    # Update portfolio
    portfolio["cash"] += sale_value
    portfolio["positions"] = other_positions + matching_positions
    
    save_json(PORTFOLIO_FILE, portfolio)
    
    print(f"✅ Sold {len(matching_positions)} position(s) in {args.market}")
    print(f"   Shares: {total_shares:.4f} @ {fmt_price(current_price)}")
    print(f"   Sale value: ${sale_value:.2f}")
    print(f"   P&L: ${pnl:+.2f}")
    print(f"   Cash: ${portfolio['cash']:.2f}")


def cmd_portfolio(args: argparse.Namespace) -> None:
    """View paper trading portfolio."""
    portfolio = load_json(PORTFOLIO_FILE, {"cash": 10000.0, "positions": []})
    
    if args.json:
        # Enrich with current prices
        result = {
            "cash": portfolio["cash"],
            "positions": [],
            "summary": {"total_value": portfolio["cash"], "total_pnl": 0.0}
        }
        
        for pos in portfolio.get("positions", []):
            if pos.get("sold"):
                continue
            
            market = resolve_market(pos["market_slug"])
            if market:
                yes, no = get_prices(market)
                cur_price = yes if pos["outcome"] == "yes" else no
            else:
                cur_price = pos["entry_price"]
            
            value = pos["shares"] * cur_price
            pnl = value - pos["cost"]
            
            pos_data = {
                "market": pos["market_question"],
                "slug": pos["market_slug"],
                "outcome": pos["outcome"],
                "shares": pos["shares"],
                "entry_price": pos["entry_price"],
                "current_price": cur_price,
                "value": value,
                "pnl": pnl
            }
            result["positions"].append(pos_data)
            result["summary"]["total_value"] += value
            result["summary"]["total_pnl"] += pnl
        
        print(json.dumps(result, indent=2))
        return
    
    # Human-readable
    print("Paper Trading Portfolio\n")
    print(f"Cash: ${portfolio['cash']:.2f}")
    
    open_positions = [p for p in portfolio.get("positions", []) if not p.get("sold")]
    
    if not open_positions:
        print("\nNo open positions")
        # Show closed positions summary if any
        closed = [p for p in portfolio.get("positions", []) if p.get("sold")]
        if closed:
            total_pnl = sum(p.get("pnl", 0) for p in closed)
            print(f"\nClosed positions: {len(closed)} (Total P&L: ${total_pnl:+.2f})")
        return
    
    total_value = portfolio["cash"]
    total_pnl = 0.0
    
    print(f"\nOpen Positions ({len(open_positions)})\n")
    print("| Market | Side | Shares | Entry | Current | Value | P&L |")
    print("|--------|------|--------|-------|---------|-------|-----|")
    
    for pos in open_positions:
        market = resolve_market(pos["market_slug"])
        if market:
            yes, no = get_prices(market)
            cur_price = yes if pos["outcome"] == "yes" else no
        else:
            cur_price = pos["entry_price"]
        
        value = pos["shares"] * cur_price
        pnl = value - pos["cost"]
        total_value += value
        total_pnl += pnl
        
        market_name = pos["market_question"]
        if len(market_name) > 30:
            market_name = market_name[:27] + "..."
        
        print(f"| {market_name} | {pos['outcome']} | {pos['shares']:.2f} | "
              f"{fmt_price(pos['entry_price'])} | {fmt_price(cur_price)} | "
              f"${value:.2f} | ${pnl:+.2f} |")
    
    print(f"\nTotal Value: ${total_value:.2f}")
    print(f"Total P&L: ${total_pnl:+.2f}")


# =============================================================================
# Smart Money Commands
# =============================================================================

def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Fetch and display Polymarket leaderboard with Smart Money scores."""
    timeframe = args.timeframe
    limit = min(args.limit, 100)
    
    # Map timeframe to API parameter
    tf_param = {"7d": "sevenDays", "30d": "thirtyDays", "all": "allTime"}.get(timeframe, "thirtyDays")
    
    data = fetch_data(f"/leaderboard?timeframe={tf_param}&limit={limit}")
    
    if data is None:
        print("Failed to fetch leaderboard", file=sys.stderr)
        sys.exit(1)
    
    if not isinstance(data, list):
        print("Invalid leaderboard data", file=sys.stderr)
        sys.exit(1)
    
    # Calculate additional metrics
    enriched_data = []
    for entry in data:
        wallet = entry.get("user", "")
        profit = float(entry.get("profit", 0))
        volume = float(entry.get("volumeTraded", 0))
        win_count = int(entry.get("wins", 0))
        loss_count = int(entry.get("losses", 0))
        total_trades = win_count + loss_count
        
        # Calculate metrics
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        roi = (profit / volume * 100) if volume > 0 else 0
        
        # Smart Money Score (0-100)
        # Factors: PnL rank (40%), Win rate (30%), ROI (20%), Volume (10%)
        pnl_score = min(100, max(0, 50 + (profit / 10000)))  # Normalize around 0
        win_score = win_rate
        roi_score = min(100, max(0, 50 + roi))
        vol_score = min(100, volume / 100000 * 100)  # $100k+ = full score
        
        smart_score = (pnl_score * 0.4 + win_score * 0.3 + roi_score * 0.2 + vol_score * 0.1)
        
        # Classify tier
        if smart_score >= 80:
            tier = "🐋 Whale"
        elif smart_score >= 60:
            tier = "🦈 Smart"
        elif smart_score >= 40:
            tier = "🐟 Active"
        else:
            tier = "🦐 Retail"
        
        enriched_data.append({
            **entry,
            "wallet": wallet,
            "profit": profit,
            "volume": volume,
            "win_rate": win_rate,
            "roi": roi,
            "smart_score": smart_score,
            "tier": tier,
            "total_trades": total_trades
        })
    
    # Sort by smart score for display
    enriched_data.sort(key=lambda x: x["smart_score"], reverse=True)
    
    if args.json:
        print(json.dumps(enriched_data, indent=2))
        return
    
    print(f"📊 Leaderboard ({timeframe}) - Sorted by Smart Money Score\n")
    print(f"{'Rank':<5} {'Wallet':<14} {'Tier':<10} {'Score':<6} {'PnL':<12} {'Win%':<6} {'ROI%':<6} {'Trades':<7}")
    print("-" * 85)
    
    for i, e in enumerate(enriched_data[:limit], 1):
        wallet_short = f"{e['wallet'][:6]}...{e['wallet'][-4:]}" if len(e['wallet']) > 12 else e['wallet']
        pnl_str = f"${e['profit']:+.0f}"
        print(f"{i:<5} {wallet_short:<14} {e['tier']:<10} {e['smart_score']:.0f}{'':<3} {pnl_str:<12} {e['win_rate']:.1f}{'':<3} {e['roi']:.1f}{'':<3} {e['total_trades']:<7}")
    
    print(f"\n💡 Use `score <wallet>` for detailed analysis of any trader")


def cmd_score(args: argparse.Namespace) -> None:
    """Score and analyze a specific wallet with validation."""
    wallet = args.wallet.lower().strip()
    if not validate_eth_address(wallet):
        print("Error: Invalid Ethereum wallet address", file=sys.stderr)
        sys.exit(1)
    
    # Fetch positions and trades
    positions = fetch_data(f"/positions?user={wallet}")
    trades = fetch_data(f"/trades?user={wallet}")
    
    if positions is None:
        print(f"Failed to fetch data for {wallet}", file=sys.stderr)
        sys.exit(1)
    
    positions = positions if isinstance(positions, list) else []
    trades = trades if isinstance(trades, list) else []
    
    # Calculate metrics
    total_pnl = 0.0
    total_volume = 0.0
    winning_positions = 0
    losing_positions = 0
    
    position_details = []
    for p in positions:
        size = float(p.get("size", 0))
        avg_price = float(p.get("avgPrice", 0))
        cur_price = float(p.get("curPrice", p.get("currentPrice", 0)))
        pnl = (cur_price - avg_price) * size
        total_pnl += pnl
        total_volume += size * avg_price
        
        if pnl > 0:
            winning_positions += 1
        elif pnl < 0:
            losing_positions += 1
        
        position_details.append({
            "market": p.get("title", p.get("market", "Unknown")),
            "outcome": p.get("outcome", "?"),
            "size": size,
            "entry": avg_price,
            "current": cur_price,
            "pnl": pnl
        })
    
    # Trading pattern analysis
    trade_count = len(trades)
    
    # HFT detection
    hft_score = 0
    if trade_count > 100:
        hft_score = min(100, trade_count / 10)
    
    # Win rate calculation
    total_positions = winning_positions + losing_positions
    win_rate = (winning_positions / total_positions * 100) if total_positions > 0 else 0
    
    # ROI
    roi = (total_pnl / total_volume * 100) if total_volume > 0 else 0
    
    # Smart Money Score components
    pnl_score = min(100, max(0, 50 + (total_pnl / 5000)))
    win_score = win_rate
    roi_score = min(100, max(0, 50 + roi * 2))
    activity_score = min(100, trade_count / 2)
    
    smart_score = (pnl_score * 0.35 + win_score * 0.25 + roi_score * 0.25 + activity_score * 0.15)
    
    # Determine trader type
    if hft_score > 70:
        trader_type = "⚡ HFT / Market Maker"
    elif smart_score >= 80:
        trader_type = "🐋 Smart Whale"
    elif smart_score >= 60:
        trader_type = "🦈 Consistent Winner"
    elif smart_score >= 40:
        trader_type = "🐟 Active Trader"
    else:
        trader_type = "🦐 Retail / New"
    
    result = {
        "wallet": wallet,
        "smart_score": round(smart_score, 1),
        "trader_type": trader_type,
        "metrics": {
            "total_pnl": total_pnl,
            "total_volume": total_volume,
            "roi_percent": roi,
            "win_rate_percent": win_rate,
            "open_positions": len(positions),
            "total_trades": trade_count,
            "winning_positions": winning_positions,
            "losing_positions": losing_positions
        },
        "component_scores": {
            "pnl_score": round(pnl_score, 1),
            "win_rate_score": round(win_score, 1),
            "roi_score": round(roi_score, 1),
            "activity_score": round(activity_score, 1)
        },
        "flags": {
            "hft_suspected": hft_score > 70,
            "high_volume": total_volume > 100000,
            "consistent_winner": win_rate > 60 and total_positions > 5
        },
        "positions": position_details[:10]  # Top 10 positions
    }
    
    if args.json:
        print(json.dumps(result, indent=2))
        return
    
    # Human-readable output
    print(f"🔍 Wallet Analysis: {wallet[:10]}...{wallet[-6:]}\n")
    
    print(f"Trader Type: {trader_type}")
    print(f"Smart Money Score: {smart_score:.1f}/100")
    print()
    
    print("Component Scores:")
    print(f"  PnL Score:    {pnl_score:.1f}/100")
    print(f"  Win Rate:     {win_score:.1f}/100")
    print(f"  ROI Score:    {roi_score:.1f}/100")
    print(f"  Activity:     {activity_score:.1f}/100")
    print()
    
    print("Key Metrics:")
    print(f"  Total P&L:    ${total_pnl:+.2f}")
    print(f"  Total Volume: ${total_volume:,.2f}")
    print(f"  ROI:          {roi:+.2f}%")
    print(f"  Win Rate:     {win_rate:.1f}% ({winning_positions}W/{losing_positions}L)")
    print(f"  Open Pos:     {len(positions)}")
    print(f"  Total Trades: {trade_count}")
    print()
    
    if result["flags"]["hft_suspected"]:
        print("⚡ Flag: High-frequency trading pattern detected")
    if result["flags"]["consistent_winner"]:
        print("✅ Flag: Consistent winner (>60% win rate)")
    if result["flags"]["high_volume"]:
        print("💰 Flag: High volume trader (>$100k)")
    
    if position_details:
        print("\nTop Positions:")
        for p in sorted(position_details, key=lambda x: abs(x["pnl"]), reverse=True)[:5]:
            emoji = "📈" if p["pnl"] > 0 else "📉" if p["pnl"] < 0 else "➖"
            print(f"  {emoji} {p['market'][:40]:<40} | {p['outcome']:<4} | P&L: ${p['pnl']:+.2f}")


def cmd_signals(args: argparse.Namespace) -> None:
    """Aggregate trading signals from smart money."""
    # Fetch leaderboard to get top traders
    leaderboard = fetch_data("/leaderboard?timeframe=thirtyDays&limit=50")
    
    if leaderboard is None or not isinstance(leaderboard, list):
        print("Failed to fetch leaderboard", file=sys.stderr)
        sys.exit(1)
    
    # Filter smart traders (top 20 by profit)
    smart_wallets = [entry.get("user") for entry in leaderboard[:20] if entry.get("user")]
    
    # Aggregate positions across smart money
    market_sentiment = {}
    
    for wallet in smart_wallets:
        positions = fetch_data(f"/positions?user={wallet}")
        if not isinstance(positions, list):
            continue
        
        for p in positions:
            market_slug = p.get("market", "")
            outcome = p.get("outcome", "")
            size = float(p.get("size", 0))
            
            if not market_slug:
                continue
            
            if market_slug not in market_sentiment:
                market_sentiment[market_slug] = {
                    "question": p.get("title", market_slug),
                    "yes_votes": 0,
                    "no_votes": 0,
                    "total_value": 0,
                    "wallets": set()
                }
            
            if outcome.lower() == "yes":
                market_sentiment[market_slug]["yes_votes"] += 1
                market_sentiment[market_slug]["total_value"] += size
            elif outcome.lower() == "no":
                market_sentiment[market_slug]["no_votes"] += 1
                market_sentiment[market_slug]["total_value"] += size
            
            market_sentiment[market_slug]["wallets"].add(wallet)
    
    # Calculate signals
    signals = []
    for slug, data in market_sentiment.items():
        total_votes = data["yes_votes"] + data["no_votes"]
        if total_votes < args.min_wallets:  # Minimum consensus threshold
            continue
        
        yes_ratio = data["yes_votes"] / total_votes
        consensus_strength = abs(yes_ratio - 0.5) * 2  # 0-1 scale
        
        if consensus_strength < 0.2:  # No clear signal
            continue
        
        signal_type = "BULLISH" if yes_ratio > 0.5 else "BEARISH"
        confidence = int(consensus_strength * 100)
        
        signals.append({
            "market": slug,
            "question": data["question"],
            "signal": signal_type,
            "confidence": confidence,
            "smart_wallets": len(data["wallets"]),
            "yes_votes": data["yes_votes"],
            "no_votes": data["no_votes"],
            "total_value": data["total_value"],
            "avg_position": data["total_value"] / total_votes if total_votes > 0 else 0
        })
    
    # Sort by confidence and wallet count
    signals.sort(key=lambda x: (x["confidence"], x["smart_wallets"]), reverse=True)
    
    if args.json:
        # Convert sets to lists for JSON serialization
        for s in signals:
            s["wallets"] = list(s.get("wallets", []))
        print(json.dumps(signals, indent=2))
        return
    
    print("📡 Smart Money Signals\n")
    print("Aggregated from top 20 traders by PnL (30d)\n")
    
    if not signals:
        print("No strong consensus signals found.")
        print(f"Tip: Try lowering --min-wallets (current: {args.min_wallets})")
        return
    
    print(f"{'Signal':<10} {'Conf':<6} {'Wallets':<8} {'Market':<45}")
    print("-" * 80)
    
    for s in signals[:args.limit]:
        emoji = "🟢" if s["signal"] == "BULLISH" else "🔴"
        print(f"{emoji} {s['signal']:<8} {s['confidence']}%{'':<3} {s['smart_wallets']:<8} {s['question'][:45]:<45}")
        print(f"   └─ YES: {s['yes_votes']} | NO: {s['no_votes']} | Avg Position: ${s['avg_position']:.0f}")
        print()
    
    print("💡 Use `event <market>` to analyze any of these markets")


# =============================================================================
# Advanced Analysis: Market Efficiency, Shapley Aggregation, Combinatorial Arbitrage
# Based on Chen & Pennock (2012), Conitzer (2009/2012), Hanson (2002/2003)
# =============================================================================

def calculate_liquidity_from_orderbook(bids: List[Dict], asks: List[Dict], mid_price: float) -> Dict:
    """
    Calculate instantaneous liquidity from orderbook data.
    Based on Chen & Pennock (2012) Definition 1: ρ_i = 1 / (∂p_i/∂q_i)
    """
    if not bids or not asks:
        return {"bid_liquidity": 0, "ask_liquidity": 0, "spread": 1.0}
    
    best_bid = float(bids[0]["price"]) if bids else 0
    best_ask = float(asks[0]["price"]) if asks else 1
    spread = best_ask - best_bid
    
    # Calculate depth-weighted liquidity
    bid_depth = sum(float(b["size"]) for b in bids)
    ask_depth = sum(float(a["size"]) for a in asks)
    
    # Estimate price impact (slope approximation)
    bid_impact = (mid_price - best_bid) / bid_depth if bid_depth > 0 else 1
    ask_impact = (best_ask - mid_price) / ask_depth if ask_depth > 0 else 1
    
    # Liquidity is inverse of price impact
    bid_liquidity = 1 / bid_impact if bid_impact > 0 else 0
    ask_liquidity = 1 / ask_impact if ask_impact > 0 else 0
    
    return {
        "bid_liquidity": round(bid_liquidity, 4),
        "ask_liquidity": round(ask_liquidity, 4),
        "spread": round(spread, 4),
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "mid_price": mid_price
    }


def calculate_worst_case_loss_lmsr(liquidity: float, n_outcomes: int = 2) -> float:
    """
    Calculate worst-case loss for LMSR market maker.
    Based on Chen & Pennock (2012) Lemma 7 and Theorem 9.
    
    For LMSR: L_max = b * H(q) where H is entropy
    Lower bound: L_min = (N-1)² * ρ / (2N²)
    """
    # Assume maximum entropy (uniform distribution) for worst case
    max_entropy = math.log(n_outcomes)
    worst_loss = liquidity * max_entropy
    
    # Calculate theoretical lower bound (Theorem 9)
    rho = liquidity / n_outcomes  # approximate instantaneous liquidity
    loss_lower_bound = ((n_outcomes - 1) ** 2) * rho / (2 * n_outcomes ** 2)
    
    return {
        "worst_case_loss": round(worst_loss, 4),
        "theoretical_lower_bound": round(loss_lower_bound, 4),
        "liquidity_parameter": liquidity
    }


def calculate_convergence_speed(price_history: List[float], window: int = 24) -> Dict:
    """
    Calculate price convergence speed (information aggregation metric).
    Based on Wolfers & Zitzewitz (2004) EMH framework.
    """
    if len(price_history) < window * 2:
        return {"convergence_speed": None, "volatility_ratio": None}
    
    # Calculate returns
    returns = [price_history[i] - price_history[i-1] for i in range(1, len(price_history))]
    
    # Volatility in early vs late period
    early_vol = np.std(returns[:window]) if len(returns) >= window else 0
    late_vol = np.std(returns[-window:]) if len(returns) >= window else 0
    
    # Convergence speed: inverse of volatility decay
    if early_vol > 0 and late_vol > 0:
        volatility_ratio = late_vol / early_vol
        convergence_speed = 1 / volatility_ratio if volatility_ratio > 0 else 0
    else:
        volatility_ratio = None
        convergence_speed = None
    
    # Price path efficiency (random walk test)
    if len(returns) > 1:
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        efficiency_score = 1 - abs(autocorr)  # Higher = more efficient (less predictable)
    else:
        efficiency_score = None
    
    return {
        "convergence_speed": round(convergence_speed, 4) if convergence_speed else None,
        "volatility_ratio": round(volatility_ratio, 4) if volatility_ratio else None,
        "efficiency_score": round(efficiency_score, 4) if efficiency_score else None,
        "early_volatility": round(early_vol, 4) if early_vol else None,
        "late_volatility": round(late_vol, 4) if late_vol else None
    }


def cmd_efficiency(args: argparse.Namespace) -> None:
    """
    Analyze market efficiency using Chen & Pennock (2012) framework.
    Calculates liquidity, worst-case loss bounds, and convergence metrics.
    """
    market = resolve_market(args.market)
    if not market:
        print(f"Error: Market not found: {args.market}", file=sys.stderr)
        sys.exit(1)
    
    market_id = get_market_id(market)
    slug = market.get("slug", args.market)
    question = market.get("question", slug)
    
    # Get current prices
    yes_price, no_price = get_prices(market)
    mid_price = (yes_price + (1 - no_price)) / 2 if yes_price and no_price else yes_price or 0.5
    
    # Fetch orderbook for liquidity analysis
    orderbook = None
    token_id = market.get("tokens", [{}])[0].get("token_id") if market.get("tokens") else None
    if token_id:
        orderbook = fetch_clob(f"/book?token_id={token_id}")
    
    # Calculate liquidity metrics
    liquidity_metrics = {}
    if orderbook and "bids" in orderbook and "asks" in orderbook:
        liquidity_metrics = calculate_liquidity_from_orderbook(
            orderbook.get("bids", []),
            orderbook.get("asks", []),
            mid_price
        )
    
    # Estimate liquidity parameter (b in LMSR)
    # From market volume and spread
    volume = float(market.get("volume", 0))
    liquidity_param = volume / 1000 if volume > 0 else 100  # heuristic
    
    # Calculate worst-case loss (LMSR framework)
    loss_metrics = calculate_worst_case_loss_lmsr(liquidity_param, n_outcomes=2)
    
    # Get historical prices for convergence analysis
    price_history = []
    if "prices" in market and isinstance(market["prices"], list):
        price_history = [float(p) for p in market["prices"]]
    
    convergence_metrics = calculate_convergence_speed(price_history, window=args.window)
    
    # Calculate information efficiency score
    info_score = 0
    if liquidity_metrics.get("spread") is not None:
        spread_score = max(0, 1 - liquidity_metrics["spread"] * 10)  # tighter spread = better
        depth_score = min(1, (liquidity_metrics.get("bid_depth", 0) + liquidity_metrics.get("ask_depth", 0)) / 10000)
        convergence_score = convergence_metrics.get("convergence_speed", 0) or 0
        efficiency_score = convergence_metrics.get("efficiency_score", 0) or 0
        
        info_score = (spread_score * 0.3 + depth_score * 0.3 + 
                     min(1, convergence_score) * 0.2 + efficiency_score * 0.2)
    
    result = {
        "market": slug,
        "question": question,
        "current_price": round(mid_price, 4),
        "liquidity": liquidity_metrics,
        "worst_case_loss": loss_metrics,
        "convergence": convergence_metrics,
        "efficiency_score": round(info_score, 4),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    if args.json:
        print(json.dumps(result, indent=2))
        return
    
    # Human-readable output
    print(f"📊 Market Efficiency Analysis: {question[:50]}")
    print(f"   Current Price: {mid_price:.2%}\n")
    
    print("💧 Liquidity Metrics (Chen & Pennock 2012):")
    if liquidity_metrics:
        print(f"  Bid Liquidity:  {liquidity_metrics.get('bid_liquidity', 'N/A')}")
        print(f"  Ask Liquidity:  {liquidity_metrics.get('ask_liquidity', 'N/A')}")
        print(f"  Spread:         {liquidity_metrics.get('spread', 'N/A'):.4f}")
        print(f"  Bid Depth:      ${liquidity_metrics.get('bid_depth', 0):,.0f}")
        print(f"  Ask Depth:      ${liquidity_metrics.get('ask_depth', 0):,.0f}")
    else:
        print("  Orderbook data unavailable")
    print()
    
    print("🛡️  Worst-Case Loss Analysis (LMSR Framework):")
    print(f"  Liquidity Param (b):  {loss_metrics['liquidity_parameter']:.2f}")
    print(f"  Worst-Case Loss:      ${loss_metrics['worst_case_loss']:,.2f}")
    print(f"  Theoretical Lower Bound: ${loss_metrics['theoretical_lower_bound']:,.2f}")
    print()
    
    print("⚡ Convergence Metrics (EMH Framework):")
    if convergence_metrics.get("convergence_speed"):
        print(f"  Convergence Speed:    {convergence_metrics['convergence_speed']:.4f}")
        print(f"  Volatility Ratio:     {convergence_metrics['volatility_ratio']:.4f}")
        print(f"  Efficiency Score:     {convergence_metrics['efficiency_score']:.4f}")
    else:
        print("  Insufficient price history for convergence analysis")
    print()
    
    print(f"📈 Overall Information Efficiency Score: {info_score:.2%}")
    
    # Interpretation
    if info_score > 0.8:
        print("   ✅ Highly efficient market (strong EMH)")
    elif info_score > 0.5:
        print("   ⚠️  Moderately efficient (some friction)")
    else:
        print("   🔴 Low efficiency (possible opportunities)")


def calculate_shapley_value(marginal_contributions: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Calculate Shapley values from marginal contributions.
    Based on Conitzer (2009/2012) Shapley Value Information Mechanism.
    """
    shapley_values = {}
    for agent, contributions in marginal_contributions.items():
        shapley_values[agent] = np.mean(contributions) if contributions else 0
    return shapley_values


def evaluate_information_value(wallet_positions: Dict, market_outcomes: Dict, 
                                scoring_rule: str = "log") -> float:
    """
    Evaluate information value using proper scoring rules.
    """
    total_score = 0
    count = 0
    
    for market_id, position in wallet_positions.items():
        if market_id not in market_outcomes:
            continue
        
        outcome = market_outcomes[market_id]
        prediction = position.get("prediction", 0.5)
        
        if scoring_rule == "log":
            # Log score (Good 1952) - protect against log(0)
            prediction = max(0.001, min(0.999, prediction))
            score = math.log(prediction) if outcome == 1 else math.log(1 - prediction)
        elif scoring_rule == "brier":
            # Brier score (Brier 1950) - bounded 0 to 1
            score = 1.0 - (prediction - outcome) ** 2
        else:
            score = 0
        
        total_score += score
        count += 1
    
    return total_score / count if count > 0 else 0


def cmd_shapley(args: argparse.Namespace) -> None:
    """
    Aggregate signals using Shapley value information mechanism.
    Based on Conitzer (2009/2012) cooperative game theory framework.
    """
    # Fetch top traders from leaderboard
    leaderboard = fetch_data("/leaderboard?timeframe=thirtyDays&limit=30")
    if not leaderboard or not isinstance(leaderboard, list):
        print("Failed to fetch leaderboard", file=sys.stderr)
        sys.exit(1)
    
    # Filter smart traders
    smart_wallets = [entry.get("user") for entry in leaderboard[:args.max_wallets] 
                     if entry.get("user")]
    
    if len(smart_wallets) < 2:
        print("Need at least 2 wallets for Shapley aggregation", file=sys.stderr)
        sys.exit(1)
    
    print(f"🎯 Shapley Value Signal Aggregation")
    print(f"   Analyzing {len(smart_wallets)} top traders...\n")
    
    # Fetch positions for each wallet
    wallet_data = {}
    for wallet in smart_wallets:
        positions = fetch_data(f"/positions?user={wallet}")
        if isinstance(positions, list):
            wallet_data[wallet] = positions
    
    # Find common markets
    market_wallets = {}
    for wallet, positions in wallet_data.items():
        for p in positions:
            market = p.get("market", "")
            if market:
                if market not in market_wallets:
                    market_wallets[market] = {}
                market_wallets[market][wallet] = {
                    "outcome": p.get("outcome", ""),
                    "size": float(p.get("size", 0)),
                    "avg_price": float(p.get("avgPrice", 0))
                }
    
    # Calculate Shapley values for each market
    shapley_results = []
    
    for market_slug, wallets in market_wallets.items():
        if len(wallets) < args.min_wallets:
            continue
        
        # Get market details
        market_info = None
        for p in wallet_data[list(wallets.keys())[0]]:
            if p.get("market") == market_slug:
                market_info = p
                break
        
        question = market_info.get("title", market_slug) if market_info else market_slug
        
        # Calculate marginal contributions
        marginal_contributions = {w: [] for w in wallets}
        
        wallet_list = list(wallets.keys())
        
        # [MAJOR FIX] Complexity protection: limit permutations to avoid n! explosion
        # For n > 7 wallets, sample permutations instead of iterating all
        MAX_PERMUTATION_WALLETS = 7
        MAX_SAMPLED_PERMUTATIONS = 5000
        
        if len(wallet_list) > MAX_PERMUTATION_WALLETS:
            # Use random sampling for large wallet sets (random already imported at top)
            print(f"   Note: Sampling {MAX_SAMPLED_PERMUTATIONS} permutations for {len(wallet_list)} wallets", 
                  file=sys.stderr)
            all_perms = []
            for _ in range(MAX_SAMPLED_PERMUTATIONS):
                all_perms.append(tuple(random.sample(wallet_list, len(wallet_list))))
        else:
            all_perms = list(permutations(wallet_list))
        
        for perm in all_perms:
            running_info = {"yes_votes": 0, "no_votes": 0, "total_size": 0.0}
            
            for wallet in perm:
                # Value before adding this wallet
                v_before = evaluate_signal_value(running_info)
                
                # Add wallet's contribution
                pos = wallets[wallet]
                running_info["total_size"] += pos["size"]
                if pos["outcome"].lower() == "yes":
                    running_info["yes_votes"] += pos["size"]
                else:
                    running_info["no_votes"] += pos["size"]
                
                # Value after
                v_after = evaluate_signal_value(running_info)
                
                marginal_contributions[wallet].append(v_after - v_before)
        
        # Calculate Shapley values
        shapley_values = calculate_shapley_value(marginal_contributions)
        
        # Calculate weighted signal
        total_shapley = sum(abs(v) for v in shapley_values.values())
        if total_shapley > 0:
            weighted_yes = sum(
                shapley_values[w] * (1 if wallets[w]["outcome"].lower() == "yes" else 0)
                for w in wallets
            ) / total_shapley
            
            consensus = abs(weighted_yes - 0.5) * 2
            signal_type = "BULLISH" if weighted_yes > 0.5 else "BEARISH"
            
            shapley_results.append({
                "market": market_slug,
                "question": question,
                "signal": signal_type,
                "confidence": round(consensus * 100, 1),
                "weighted_yes": round(weighted_yes, 4),
                "shapley_values": {w[:10] + "...": round(v, 4) for w, v in shapley_values.items()},
                "participating_wallets": len(wallets)
            })
    
    # Sort by confidence
    shapley_results.sort(key=lambda x: x["confidence"], reverse=True)
    
    if args.json:
        print(json.dumps(shapley_results[:args.limit], indent=2))
        return
    
    print(f"📊 Shapley-Weighted Smart Money Signals\n")
    
    if not shapley_results:
        print("No significant Shapley signals found.")
        return
    
    print(f"{'Signal':<10} {'Conf':<8} {'Wallets':<8} {'Market':<40}")
    print("-" * 75)
    
    for r in shapley_results[:args.limit]:
        emoji = "🟢" if r["signal"] == "BULLISH" else "🔴"
        print(f"{emoji} {r['signal']:<8} {r['confidence']}%{'':<2} {r['participating_wallets']:<8} {r['question'][:40]:<40}")
        print(f"   Weighted YES: {r['weighted_yes']:.2%}")
        print()
    
    print("💡 Shapley values ensure fair credit allocation based on marginal contributions")


def evaluate_signal_value(info: Dict) -> float:
    """Helper function to evaluate signal information value."""
    total = info.get("total_size", 0)
    if total <= 0:
        return 0.0
    
    yes_votes = info.get("yes_votes", 0)
    no_votes = info.get("no_votes", 0)
    
    # Avoid division by zero and ensure valid ratios
    if yes_votes <= 0 or no_votes <= 0:
        return 0.0
    
    # Entropy-based information measure
    yes_ratio = yes_votes / total
    no_ratio = no_votes / total
    
    # Clamp to avoid log(0) issues
    yes_ratio = max(0.001, min(0.999, yes_ratio))
    no_ratio = 1.0 - yes_ratio
    
    # Shannon entropy
    entropy = -(yes_ratio * math.log(yes_ratio) + no_ratio * math.log(no_ratio))
    max_entropy = math.log(2)
    
    # Information = 1 - normalized entropy (higher = more consensus)
    information = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
    return information * total  # Scale by total size


def calculate_conditional_probability(p_a: float, p_b: float, p_a_and_b: float) -> Tuple[float, float]:
    """
    Calculate P(A|B) and P(B|A) from joint and marginal probabilities.
    Based on Hanson (2002/2003) combinatorial prediction market framework.
    """
    # [MAJOR FIX] Protect against division by zero
    p_a_given_b = p_a_and_b / p_b if p_b > 0.001 else 0.0
    p_b_given_a = p_a_and_b / p_a if p_a > 0.001 else 0.0
    return p_a_given_b, p_b_given_a


def detect_arbitrage_opportunity(market_a_price: float, market_b_price: float,
                                  joint_price: float, threshold: float = 0.02) -> Dict:
    """
    Detect arbitrage between related markets.
    Checks: P(A|B) * P(B) = P(A ∧ B) consistency.
    """
    # [MAJOR FIX] Validate price inputs to avoid invalid calculations
    if (market_a_price <= 0 or market_a_price >= 1 or
        market_b_price <= 0 or market_b_price >= 1 or
        joint_price <= 0 or joint_price >= 1):
        return {"exists": False, "error": "invalid_price"}
    
    # Theoretical joint probability
    theoretical_joint = market_a_price * market_b_price
    
    # Check consistency
    diff = abs(joint_price - theoretical_joint)
    
    if diff > threshold:
        if joint_price > theoretical_joint:
            return {
                "exists": True,
                "type": "buy_components",
                "profit_potential": round(diff, 4),
                "theoretical": round(theoretical_joint, 4),
                "actual": round(joint_price, 4),
                "strategy": f"Buy A and B separately, sell joint"
            }
        else:
            return {
                "exists": True,
                "type": "buy_joint",
                "profit_potential": round(diff, 4),
                "theoretical": round(theoretical_joint, 4),
                "actual": round(joint_price, 4),
                "strategy": f"Buy joint, sell A and B separately"
            }
    
    return {"exists": False, "difference": round(diff, 4)}


def cmd_comboarb(args: argparse.Namespace) -> None:
    """
    Detect combinatorial arbitrage opportunities across related markets.
    Based on Hanson (2002/2003) modular prediction market framework.
    """
    # Fetch trending markets
    markets = fetch_data("/markets?limit=100&active=true")
    if not markets or not isinstance(markets, list):
        print("Failed to fetch markets", file=sys.stderr)
        sys.exit(1)
    
    print(f"🔍 Combinatorial Arbitrage Scanner")
    print(f"   Analyzing {len(markets)} markets for related pairs...\n")
    
    # Find potentially related markets (simple heuristic: similar keywords)
    market_pairs = []
    keywords_map = {}
    
    for m in markets:
        question = m.get("question", "").lower()
        slug = m.get("slug", "")
        
        # Extract keywords
        words = set(question.split()) - {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "will", "by"}
        
        for word in words:
            if len(word) > 3:  # Only significant words
                if word not in keywords_map:
                    keywords_map[word] = []
                keywords_map[word].append(m)
    
    # Find pairs with common keywords
    for word, market_list in keywords_map.items():
        if len(market_list) >= 2:
            for i in range(len(market_list)):
                for j in range(i+1, len(market_list)):
                    m1, m2 = market_list[i], market_list[j]
                    if m1["slug"] != m2["slug"]:
                        market_pairs.append((m1, m2, word))
    
    # Limit pairs to analyze
    market_pairs = market_pairs[:50]
    
    # Analyze each pair
    arbitrage_opportunities = []
    
    for m1, m2, keyword in market_pairs:
        p1, _ = get_prices(m1)
        p2, _ = get_prices(m2)
        
        if not p1 or not p2:
            continue
        
        # Look for joint market (simplified heuristic)
        # In practice, would search for markets with both conditions
        joint_market = None
        for m in markets:
            q = m.get("question", "").lower()
            if keyword in q and (m1["question"].split()[0] in q or m2["question"].split()[0] in q):
                if m["slug"] != m1["slug"] and m["slug"] != m2["slug"]:
                    joint_market = m
                    break
        
        if joint_market:
            p_joint, _ = get_prices(joint_market)
            if p_joint:
                arb = detect_arbitrage_opportunity(p1, p2, p_joint, threshold=args.threshold)
                
                if arb.get("exists"):
                    arbitrage_opportunities.append({
                        "market_a": m1.get("question", m1["slug"])[:40],
                        "market_b": m2.get("question", m2["slug"])[:40],
                        "joint_market": joint_market.get("question", joint_market["slug"])[:40],
                        "common_keyword": keyword,
                        "p_a": round(p1, 4),
                        "p_b": round(p2, 4),
                        "p_joint": round(p_joint, 4),
                        **arb
                    })
    
    # Sort by profit potential
    arbitrage_opportunities.sort(key=lambda x: x.get("profit_potential", 0), reverse=True)
    
    if args.json:
        print(json.dumps(arbitrage_opportunities[:args.limit], indent=2))
        return
    
    print(f"📈 Combinatorial Arbitrage Opportunities (threshold: {args.threshold:.1%})\n")
    
    if not arbitrage_opportunities:
        print("No significant arbitrage opportunities found.")
        print("Markets appear consistent with modular probability framework.")
        return
    
    for i, arb in enumerate(arbitrage_opportunities[:args.limit], 1):
        print(f"{i}. 🎯 {arb['type'].replace('_', ' ').title()}")
        print(f"   Profit Potential: {arb['profit_potential']:.2%}")
        print(f"   Market A: {arb['market_a']} ({arb['p_a']:.2%})")
        print(f"   Market B: {arb['market_b']} ({arb['p_b']:.2%})")
        print(f"   Joint:    {arb['joint_market']} ({arb['p_joint']:.2%})")
        print(f"   Theoretical Joint: {arb['theoretical']:.2%}")
        print(f"   Common Keyword: '{arb['common_keyword']}'")
        print()
    
    print("💡 Based on Hanson (2002/2003) modular prediction market framework")
    print("   In efficient markets: P(A|B) * P(B) = P(A ∧ B)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Unified Toolkit v1.4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Browse
  %(prog)s trending
  %(prog)s search "trump"
  %(prog)s movers --timeframe 1w
  
  # NEW in v1.3.0 - Tags, Sports, CLOB
  %(prog)s tags                    # List categories
  %(prog)s sports                  # List sports leagues
  %(prog)s price <token_id>        # Get CLOB price
  %(prog)s book <token_id>         # Get orderbook
  
  # Analyze
  %(prog)s event trump-2028
  %(prog)s analyze trump-2028
  %(prog)s monitor trump-2028
  
  # Watchlist
  %(prog)s watch add trump-2028 --alert-at 60
  %(prog)s watch list
  %(prog)s alerts
  
  # Paper trading
  %(prog)s buy trump-2028 100 --outcome yes
  %(prog)s sell trump-2028
  %(prog)s portfolio
  
  # Smart Money
  %(prog)s leaderboard --timeframe 30d
  %(prog)s score 0x1234567890abcdef1234567890abcdef12345678
  %(prog)s signals --min-wallets 5
  
  # Profile
  %(prog)s profile 0x1234567890abcdef1234567890abcdef12345678
  
  # NEW in v1.4.0 - Advanced Analysis
  %(prog)s efficiency trump-2028   # Market efficiency analysis (Chen & Pennock)
  %(prog)s shapley                 # Shapley value signal aggregation (Conitzer)
  %(prog)s comboarb                # Combinatorial arbitrage detection (Hanson)

For cron jobs, use --json for machine-readable output and check exit codes.
        """
    )
    
    parser.add_argument("--version", action="version", version="%(prog)s 1.4.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # --- Browse Commands ---
    
    p_trending = subparsers.add_parser("trending", help="Trending markets by 24h volume")
    p_trending.add_argument("--limit", type=int, default=100, help="Number of markets (max 100)")
    p_trending.add_argument("--compact", action="store_true", help="Compact output")
    p_trending.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_search = subparsers.add_parser("search", help="Search markets (uses public-search API)")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, default=100, help="Number of results")
    p_search.add_argument("--compact", action="store_true", help="Compact output")
    p_search.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_movers = subparsers.add_parser("movers", help="Biggest price movers")
    p_movers.add_argument("--timeframe", choices=["24h", "1w", "1m"], default="24h", help="Time period")
    p_movers.add_argument("--min-volume", type=float, default=10, help="Minimum volume in $K")
    p_movers.add_argument("--limit", type=int, default=10, help="Number of results")
    p_movers.add_argument("--compact", action="store_true", help="Compact output")
    p_movers.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_event = subparsers.add_parser("event", help="Get market details")
    p_event.add_argument("market", help="Market URL, slug, or ID")
    p_event.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- NEW in v1.3.0 - Tags and Sports ---
    
    p_tags = subparsers.add_parser("tags", help="List available market categories")
    p_tags.add_argument("--limit", type=int, default=50, help="Number of tags")
    p_tags.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_sports = subparsers.add_parser("sports", help="List sports leagues")
    p_sports.add_argument("--limit", type=int, default=30, help="Number of leagues")
    p_sports.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- NEW in v1.3.0 - CLOB Commands ---
    
    p_price = subparsers.add_parser("price", help="Get current price from CLOB")
    p_price.add_argument("token_id", help="Token ID")
    p_price.add_argument("--side", choices=["buy", "sell"], default="buy", help="Side")
    p_price.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_book = subparsers.add_parser("book", help="Get orderbook from CLOB")
    p_book.add_argument("token_id", help="Token ID")
    p_book.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- Analysis Commands ---
    
    p_analyze = subparsers.add_parser("analyze", help="Analyze market for trading edges")
    p_analyze.add_argument("market", help="Market URL, slug, or ID")
    p_analyze.add_argument("--arb-threshold", type=float, default=DEFAULT_ARB_USD_THRESHOLD,
                          help=f"Arbitrage threshold (default: {DEFAULT_ARB_USD_THRESHOLD})")
    p_analyze.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_monitor = subparsers.add_parser("monitor", help="Monitor market for alerts (exit code = alert count)")
    p_monitor.add_argument("market", help="Market URL, slug, or ID")
    p_monitor.add_argument("--price-threshold", type=float, default=DEFAULT_PRICE_PCT_THRESHOLD,
                          help=f"Price change %% threshold (default: {DEFAULT_PRICE_PCT_THRESHOLD})")
    p_monitor.add_argument("--whale-threshold", type=float, default=DEFAULT_WHALE_USD_THRESHOLD,
                          help=f"Volume spike $ threshold (default: {DEFAULT_WHALE_USD_THRESHOLD})")
    p_monitor.add_argument("--arb-threshold", type=float, default=DEFAULT_ARB_USD_THRESHOLD,
                          help=f"Arbitrage $ threshold (default: {DEFAULT_ARB_USD_THRESHOLD})")
    p_monitor.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_profile = subparsers.add_parser("profile", help="Fetch user profile")
    p_profile.add_argument("wallet", help="Wallet address (0x...)")
    p_profile.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- Watchlist Commands ---
    
    p_watch = subparsers.add_parser("watch", help="Manage watchlist")
    p_watch.add_argument("action", choices=["list", "add", "remove"], help="Action to perform")
    p_watch.add_argument("market", nargs="?", help="Market slug/URL (for add/remove)")
    p_watch.add_argument("--alert-at", type=float, metavar="PCT",
                        help="Alert when price reaches this %% (e.g., 60 for 60%%)")
    p_watch.add_argument("--alert-change", type=float, metavar="PCT",
                        help="Alert on %% price change from previous check")
    p_watch.add_argument("--json", action="store_true", help="Output as JSON (for list)")
    
    p_alerts = subparsers.add_parser("alerts", help="Check watchlist for alerts")
    p_alerts.add_argument("--quiet", action="store_true", help="Only output if triggered")
    p_alerts.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- Paper Trading Commands ---
    
    p_buy = subparsers.add_parser("buy", help="Paper trade: buy position")
    p_buy.add_argument("market", help="Market URL or slug")
    p_buy.add_argument("amount", type=float, help="Amount in USD")
    p_buy.add_argument("--outcome", choices=["yes", "no"], default="yes", help="Outcome to buy")
    
    p_sell = subparsers.add_parser("sell", help="Paper trade: sell position")
    p_sell.add_argument("market", help="Market URL or slug to sell")
    
    p_portfolio = subparsers.add_parser("portfolio", help="View paper portfolio")
    p_portfolio.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- Smart Money Commands ---
    
    p_leaderboard = subparsers.add_parser("leaderboard", help="View trader leaderboard with Smart Money scores")
    p_leaderboard.add_argument("--timeframe", choices=["7d", "30d", "all"], default="30d", help="Time period")
    p_leaderboard.add_argument("--limit", type=int, default=20, help="Number of traders (max 100)")
    p_leaderboard.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_score = subparsers.add_parser("score", help="Analyze and score a specific wallet")
    p_score.add_argument("wallet", help="Wallet address (0x...)")
    p_score.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_signals = subparsers.add_parser("signals", help="Aggregate smart money signals")
    p_signals.add_argument("--min-wallets", type=int, default=3, help="Minimum smart wallets for signal")
    p_signals.add_argument("--limit", type=int, default=10, help="Number of signals to show")
    p_signals.add_argument("--json", action="store_true", help="Output as JSON")
    
    # --- NEW: Advanced Analysis Commands (v1.4.0) ---
    
    p_efficiency = subparsers.add_parser("efficiency", help="Analyze market efficiency (Chen & Pennock 2012)")
    p_efficiency.add_argument("market", help="Market URL, slug, or ID")
    p_efficiency.add_argument("--window", type=int, default=24, help="Window size for convergence analysis")
    p_efficiency.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_shapley = subparsers.add_parser("shapley", help="Shapley value signal aggregation (Conitzer 2012)")
    p_shapley.add_argument("--max-wallets", type=int, default=10, help="Maximum wallets to analyze")
    p_shapley.add_argument("--min-wallets", type=int, default=3, help="Minimum wallets per market")
    p_shapley.add_argument("--limit", type=int, default=10, help="Number of signals to show")
    p_shapley.add_argument("--json", action="store_true", help="Output as JSON")
    
    p_comboarb = subparsers.add_parser("comboarb", help="Detect combinatorial arbitrage (Hanson 2002/2003)")
    p_comboarb.add_argument("--threshold", type=float, default=0.02, help="Arbitrage detection threshold")
    p_comboarb.add_argument("--limit", type=int, default=10, help="Number of opportunities to show")
    p_comboarb.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to command
    commands = {
        "trending": cmd_trending,
        "search": cmd_search,
        "movers": cmd_movers,
        "event": cmd_event,
        "tags": cmd_tags,
        "sports": cmd_sports,
        "price": cmd_price,
        "book": cmd_book,
        "analyze": cmd_analyze,
        "monitor": cmd_monitor,
        "profile": cmd_profile,
        "watch": cmd_watch,
        "alerts": cmd_alerts,
        "buy": cmd_buy,
        "sell": cmd_sell,
        "portfolio": cmd_portfolio,
        "leaderboard": cmd_leaderboard,
        "score": cmd_score,
        "signals": cmd_signals,
        "efficiency": cmd_efficiency,
        "shapley": cmd_shapley,
        "comboarb": cmd_comboarb,
    }
    
    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
