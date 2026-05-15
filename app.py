from flask import Flask, render_template, request, jsonify, send_from_directory
import math
import os
import time
import unicodedata
from itertools import product, combinations
from concurrent.futures import ThreadPoolExecutor
import uuid
import threading
from datetime import datetime, timezone, timedelta
import requests as _req

app = Flask(__name__)
SAVED_MATCHES = []


def _espn_get(url, params=None, timeout=8, retries=2, delay=1.0):
    """GET with retry: on network error or 5xx, waits `delay` seconds and retries."""
    for attempt in range(retries + 1):
        try:
            r = _req.get(url, params=params, timeout=timeout)
            if r.status_code < 500:
                return r
            if attempt < retries:
                time.sleep(delay)
        except (_req.exceptions.ConnectionError, _req.exceptions.Timeout):
            if attempt < retries:
                time.sleep(delay)
            else:
                raise
    return r

# ─── Constants ────────────────────────────────────────────────────────────────
AVG_HOME_GOALS        = 1.49   # Historical football home average
AVG_AWAY_GOALS        = 1.12   # Historical football away average
LEAGUE_GOALS_PER_GAME = 1.35   # League average goals scored per team per game
DC_RHO                = -0.13  # Dixon-Coles low-score correction factor
MAX_GOALS             = 7

# ─── League-specific goal averages ───────────────────────────────────────────
# (avg_home_goals, avg_away_goals, league_avg_per_team)
_LEAGUE_PARAMS = {
    'premier league':   (1.53, 1.16, 1.38),
    'bundesliga':       (1.64, 1.26, 1.46),
    'la liga':          (1.47, 1.12, 1.31),
    'serie a':          (1.38, 1.05, 1.24),
    'ligue 1':          (1.42, 1.08, 1.27),
    'champions league': (1.45, 1.10, 1.30),
    'europa league':    (1.55, 1.20, 1.38),
    'libertadores':     (1.41, 1.08, 1.26),
    'eredivisie':       (1.72, 1.35, 1.55),
    'primeira liga':    (1.44, 1.09, 1.29),
}

def _league_consts(league_name):
    lg = (league_name or '').lower()
    for key, vals in _LEAGUE_PARAMS.items():
        if key in lg:
            return vals
    return AVG_HOME_GOALS, AVG_AWAY_GOALS, LEAGUE_GOALS_PER_GAME

# ─── Poisson + Dixon-Coles ────────────────────────────────────────────────────

def poisson_pmf(lmbd, k):
    if lmbd <= 0 or k < 0:
        return 0.0
    return (lmbd ** k) * math.exp(-lmbd) / math.factorial(k)

def dc_tau(h, a, lh, la, rho=DC_RHO):
    """Dixon-Coles correction for {0-0, 1-0, 0-1, 1-1} scoreline bias."""
    if   h == 0 and a == 0: return max(1.0 - lh * la * rho, 1e-6)
    elif h == 1 and a == 0: return 1.0 + la * rho
    elif h == 0 and a == 1: return 1.0 + lh * rho
    elif h == 1 and a == 1: return 1.0 - rho
    return 1.0

def match_probabilities(lh, la):
    """Full scoreline matrix with Dixon-Coles correction and renormalization."""
    raw = {}
    for h, a in product(range(MAX_GOALS + 1), repeat=2):
        p = poisson_pmf(lh, h) * poisson_pmf(la, a) * dc_tau(h, a, lh, la)
        raw[(h, a)] = max(p, 0.0)

    total = sum(raw.values()) or 1.0
    probs = {k: v / total for k, v in raw.items()}

    p1 = sum(p for (h, a), p in probs.items() if h > a)
    px = sum(p for (h, a), p in probs.items() if h == a)
    p2 = sum(p for (h, a), p in probs.items() if h < a)
    return probs, p1, px, p2

# ─── Lambda Calculation ───────────────────────────────────────────────────────

def calculate_lambdas(hs, hc, as_, ac, ng=5, league=''):
    """Strength-index model using league-specific goal averages for normalization."""
    avg_home, avg_away, L = _league_consts(league)
    games = max(ng, 1)
    hs_pg = hs / games; hc_pg = hc / games
    as_pg = as_ / games; ac_pg = ac / games
    home_attack  = hs_pg / avg_home
    away_defense = ac_pg / avg_home
    away_attack  = as_pg / avg_away
    home_defense = hc_pg / avg_away
    lh = home_attack * away_defense * avg_home
    la = away_attack * home_defense * avg_away
    return max(min(lh, 4.5), 0.50), max(min(la, 4.5), 0.50)

def form_adjustment(lmbd, form_pts):
    """±20% max. form_pts 0–15 (5 games × 3 pts). 7.5 = neutral."""
    factor = 1.0 + (form_pts - 7.5) / 37.5
    return lmbd * max(factor, 0.50)

# ─── Market & EV ─────────────────────────────────────────────────────────────

def calc_vig(oh, ox, oa):
    return 1.0 / oh + 1.0 / ox + 1.0 / oa - 1.0

def fair_prob(odd, vig):
    return (1.0 / odd) / (1.0 + vig)

def hybrid_prob(model_p, market_p):
    """
    Dynamic blend: model weight 45–60% based on model/market agreement.
    High divergence → trust market more (it may have information we lack).
    """
    divergence = abs(model_p - market_p) / max(model_p + market_p, 0.01)
    agreement  = 1.0 - min(divergence * 2.0, 1.0)
    model_w    = 0.45 + agreement * 0.15
    return model_p * model_w + market_p * (1.0 - model_w)

def calc_ev(prob, odd):
    return prob * odd - 1.0

def kelly_fraction(prob, odd):
    if odd <= 1.0:
        return 0.0
    return max((prob * odd - 1.0) / (odd - 1.0), 0.0)

def dynamic_kelly(k, conf):
    """Fractional Kelly scaled by confidence tier."""
    if conf < 40: return k * 0.35
    if conf < 55: return k * 0.55
    if conf < 70: return k * 0.75
    return k * 1.00

# ─── Confidence & Grading ─────────────────────────────────────────────────────

def confidence_score(ev_val, model_p, market_p, k_raw, odd=2.0):
    """
    Principled 0–100 confidence with strong variance penalty.

    Components:
      EV quality  (35 pts): edge size, saturates at EV=12%
      Value ratio (30 pts): model_p / market_p > 1 = value found
      Kelly size  (20 pts): large Kelly = strong edge signal
      Base prob   (15 pts): rewards high-prob (low-odds) events

    Variance penalty (log-scale, subtracted):
      Odds 2.5 → −9  |  3.0 → −14  |  4.0 → −24  |  6.0 → −39  |  8.0+ → −45 cap
    High-odds bets get penalised because a ±5% model error has much larger
    impact on a 20% estimate than on a 65% estimate.
    """
    vr        = model_p / max(market_p, 0.01)
    ev_pts    = min(max(ev_val / 0.12, 0.0), 1.0) * 35
    vr_pts    = min(max((vr - 1.0) / 0.35, 0.0), 1.0) * 30
    kelly_pts = min(k_raw / 0.10, 1.0) * 20
    prob_pts  = min(model_p / 0.70, 1.0) * 15
    raw = ev_pts + vr_pts + kelly_pts + prob_pts

    variance_pen = min(max(math.log(odd / 2.0), 0.0) * 35.0, 45.0)

    return min(max(raw - variance_pen, 0.0), 100.0)

_GRADE_C_WHITELIST = {'ТБ2.5', 'ОЗ Да'}
_DISABLED_MARKETS  = {'ТБ3.5'}  # historically −ROI, disabled
_GRADE_A_ONLY     = {'ТМ2.5', 'ОЗ Нет', 'X2'}  # historically −ROI at B; require Grade A

def bet_grade(conf, ev_val, name=''):
    if conf >= 68 and ev_val >= 0.08: return "A"
    if name in _GRADE_A_ONLY: return "D"
    if conf >= 52 and ev_val >= 0.05: return "B"
    # Relaxed B for historically profitable over/btts markets
    if name in _GRADE_C_WHITELIST and conf >= 44 and ev_val >= 0.035: return "B"
    # Grade C only for whitelisted markets — avoids short-odds traps
    if name in _GRADE_C_WHITELIST and conf >= 35 and ev_val >= 0.02: return "C"
    return "D"

# ─── ESPN team stats helper ───────────────────────────────────────────────────

def get_team_season(team_id, side='total'):
    r = _espn_get(
        f'https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{team_id}'
    )
    r.raise_for_status()
    items = r.json().get('team', {}).get('record', {}).get('items', [])
    def parse(t):
        item = next((i for i in items if i.get('type') == t), None)
        if not item: return None
        stats = {s['name']: s['value'] for s in item.get('stats', [])}
        gp = int(stats.get('gamesPlayed', 0) or 0)
        return int(stats.get('pointsFor', 0) or 0), int(stats.get('pointsAgainst', 0) or 0), gp
    split = parse(side)
    total = parse('total')
    if split and split[2] >= 3:
        return split
    if total and total[2]:
        return total
    return 0, 0, 5


def get_h2h_factor(home_id, away_id, max_meetings=10):
    """Return (lh_factor, la_factor, count) based on H2H history via ESPN schedule.
    Factors are in [0.92, 1.08]; returns (1.0, 1.0, 0) when data is insufficient.
    home_wins/away_wins are relative to home_id being the 'home' team in today's match.
    """
    if not home_id or not away_id:
        return 1.0, 1.0, 0
    cache_key = (str(home_id), str(away_id))
    cached = _H2H_CACHE.get(cache_key)
    if cached and time.time() - cached[3] < _H2H_TTL:
        return cached[0], cached[1], cached[2]
    try:
        r = _espn_get(
            f'https://site.api.espn.com/apis/site/v2/sports/soccer/all/teams/{home_id}/schedule',
            timeout=6
        )
        if r.status_code != 200:
            return 1.0, 1.0, 0
        events = r.json().get('events', [])
        home_wins = draws = away_wins = 0
        count = 0
        for ev in events:
            if count >= max_meetings:
                break
            comp = (ev.get('competitions') or [{}])[0]
            if not isinstance(comp, dict):
                continue
            if _g(comp, 'status', 'type', 'state') != 'post':
                continue
            cs = comp.get('competitors') or []
            if len(cs) < 2:
                continue
            ids_in_match = [str((c.get('team') or {}).get('id', '')) for c in cs]
            if str(away_id) not in ids_in_match:
                continue
            hc = next((c for c in cs if c.get('homeAway') == 'home'), cs[0])
            ac = next((c for c in cs if c.get('homeAway') == 'away'), cs[1])
            try:
                hs = int(hc.get('score') or 0)
                as_ = int(ac.get('score') or 0)
            except Exception:
                continue
            # Determine result from today's home_id perspective
            if str((hc.get('team') or {}).get('id', '')) == str(home_id):
                if hs > as_:   home_wins += 1
                elif hs == as_: draws += 1
                else:           away_wins += 1
            else:
                if as_ > hs:   home_wins += 1
                elif hs == as_: draws += 1
                else:           away_wins += 1
            count += 1
        if count < 3:
            return 1.0, 1.0, count
        neutral = 0.40
        lh_factor = round(max(min(1.0 + (home_wins / count - neutral) * 0.24, 1.08), 0.92), 3)
        la_factor = round(max(min(1.0 + (away_wins / count - neutral) * 0.24, 1.08), 0.92), 3)
        _H2H_CACHE[cache_key] = (lh_factor, la_factor, count, time.time())
        return lh_factor, la_factor, count
    except Exception:
        return 1.0, 1.0, 0


# ─── Core analysis helper ─────────────────────────────────────────────────────

def _run_analysis(home_team, away_team, hs, hc, as_, ac, fh, fa, ng, odds, league='', h2h=(1.0, 1.0)):
    """Run full Poisson + EV analysis. Returns match_data dict ready for SAVED_MATCHES."""
    if not (odds.get('oh') and odds.get('ox') and odds.get('oa')):
        return {'id': str(uuid.uuid4()), 'home': home_team, 'away': away_team,
                'lh': 0, 'la': 0, 'vig': 0, 'best': None, 'top3': []}

    def _o(key, default=2.0):
        val = odds.get(key)
        try: return max(float(val), 1.01) if val else default
        except: return default
    def _oopt(key):
        val = odds.get(key)
        try: return max(float(val), 1.01) if val else None
        except: return None

    oh, ox, oa   = _o('oh'), _o('ox'), _o('oa')
    o1x, ox2     = _o('o1x', 1.5), _o('ox2', 1.5)
    otb          = _oopt('otb');    otm    = _oopt('otm')
    otm35  = _oopt('otm35')
    ob_yes = _oopt('ob_yes'); ob_no  = _oopt('ob_no')

    lh, la = calculate_lambdas(hs, hc, as_, ac, ng, league)
    lh = form_adjustment(lh, fh)
    la = form_adjustment(la, fa)
    lh = round(max(min(lh * h2h[0], 4.5), 0.50), 4)
    la = round(max(min(la * h2h[1], 4.5), 0.50), 4)

    probs, p1, px, p2 = match_probabilities(lh, la)
    vig = calc_vig(oh, ox, oa)
    def mp(odd): return fair_prob(odd, vig)

    markets = [
        ('П1',  p1,       oh,  mp(oh)),
        ('X',   px,       ox,  mp(ox)),
        ('П2',  p2,       oa,  mp(oa)),
        ('1X',  p1 + px,  o1x, mp(oh) + mp(ox)),
        ('X2',  px + p2,  ox2, mp(ox) + mp(oa)),
    ]
    p_tb25     = sum(p for (h, a), p in probs.items() if h + a >= 3)
    p_tb35     = sum(p for (h, a), p in probs.items() if h + a >= 4)
    p_btts_yes = sum(p for (h, a), p in probs.items() if h > 0 and a > 0)
    if otb and otm:
        vig_ov = 1/otb + 1/otm - 1
        markets.append(('ТБ2.5', p_tb25,       otb, fair_prob(otb, vig_ov)))
        markets.append(('ТМ2.5', 1.0 - p_tb25, otm, fair_prob(otm, vig_ov)))
    elif otb:  markets.append(('ТБ2.5', p_tb25,       otb, mp(otb)))
    elif otm:  markets.append(('ТМ2.5', 1.0 - p_tb25, otm, mp(otm)))
    if otm35:  markets.append(('ТМ3.5', 1.0 - p_tb35, otm35, mp(otm35)))
    if ob_yes and ob_no:
        vig_btts = 1/ob_yes + 1/ob_no - 1
        markets.append(('ОЗ Да',  p_btts_yes,       ob_yes, fair_prob(ob_yes, vig_btts)))
        markets.append(('ОЗ Нет', 1.0 - p_btts_yes, ob_no,  fair_prob(ob_no,  vig_btts)))
    elif ob_yes: markets.append(('ОЗ Да',  p_btts_yes,       ob_yes, mp(ob_yes)))
    elif ob_no:  markets.append(('ОЗ Нет', 1.0 - p_btts_yes, ob_no,  mp(ob_no)))

    bets = {}
    for name, model_p, odd, market_p in markets:
        if name in _DISABLED_MARKETS:
            continue
        if name == '1X' and odd < 1.40:
            continue
        h_prob = hybrid_prob(model_p, market_p)
        ev_val = calc_ev(h_prob, odd)
        if ev_val <= 0:
            continue
        k_raw  = kelly_fraction(h_prob, odd)
        conf   = confidence_score(ev_val, model_p, market_p, k_raw, odd)
        k_dyn  = dynamic_kelly(k_raw, conf)
        grade  = bet_grade(conf, ev_val, name)
        if grade in ('C', 'D'):
            continue
        vr     = model_p / max(market_p, 0.01)
        bets[name] = {
            'model_prob':  round(model_p  * 100, 1),
            'market_prob': round(market_p * 100, 1),
            'prob':        round(h_prob   * 100, 1),
            'odd':         round(odd, 2),
            'ev':          round(ev_val   * 100, 1),
            'kelly':       round(k_dyn    * 100, 2),
            'conf':        round(conf,     1),
            'grade':       grade,
            'vr':          round(vr, 2),
            'logical':     round(ev_val * (conf / 100), 4),
        }

    ranked = sorted(bets.items(), key=lambda x: x[1]['logical'], reverse=True)
    seen_dc = False
    top3 = []
    for item in ranked:
        if item[0] in ('1X', 'X2'):
            if seen_dc: continue
            seen_dc = True
        top3.append(item)
        if len(top3) == 3: break

    best = top3[0] if top3 else None
    return {
        'id':   str(uuid.uuid4()),
        'home': home_team,
        'away': away_team,
        'lh':   round(lh, 2),
        'la':   round(la, 2),
        'vig':  round(vig * 100, 2),
        'best': {'name': best[0], **best[1]} if best else None,
        'top3': top3,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sw.js")
def service_worker():
    return send_from_directory("static", "sw.js", mimetype="application/javascript")


@app.route("/restore", methods=["POST"])
def restore():
    """Re-sync server SAVED_MATCHES from LocalStorage data sent on page load."""
    incoming = request.json.get("matches", [])
    SAVED_MATCHES.clear()
    SAVED_MATCHES.extend(incoming)
    return jsonify({"ok": True, "count": len(SAVED_MATCHES)})


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        d         = request.json
        home_team = (d.get("home_team") or "Хозяева").strip()
        away_team = (d.get("away_team") or "Гости").strip()
        hs  = max(int(d.get("hs",  0)), 0)
        hc  = max(int(d.get("hc",  0)), 0)
        as_ = max(int(d.get("as",  0)), 0)
        ac  = max(int(d.get("ac",  0)), 0)
        fh  = max(min(float(d.get("fh", 7.5)), 15.0), 0.0)
        fa  = max(min(float(d.get("fa", 7.5)), 15.0), 0.0)
        ng  = max(int(d.get("ng", 5)), 1)
        league = d.get("league", "")
        h2h_raw = d.get("h2h") or {}
        try:    h2h = (float(h2h_raw.get('lh', 1.0)), float(h2h_raw.get('la', 1.0)))
        except: h2h = (1.0, 1.0)
        match_data = _run_analysis(home_team, away_team, hs, hc, as_, ac, fh, fa, ng,
                                   d.get("odds", {}), league, h2h)
        SAVED_MATCHES.append(match_data)
        return jsonify({"top3": match_data["top3"], "saved_matches": SAVED_MATCHES,
                        "match": match_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/clear", methods=["POST"])
def clear():
    SAVED_MATCHES.clear()
    return jsonify({"ok": True})


@app.route("/build_express", methods=["POST"])
def build_express():
    selected   = request.json.get("match_ids", [])
    candidates = [m for m in SAVED_MATCHES if m["id"] in selected and m.get("best")]

    if len(candidates) < 2:
        return jsonify({"error": "Нужно минимум 2 матча"})

    GRADE_W = {"A": 4, "B": 3, "C": 2, "D": 1}
    combos  = []

    for r in [2, 3]:
        for combo in combinations(candidates, r):
            tot_odd, tot_prob = 1.0, 1.0
            grades = []
            for m in combo:
                tot_odd  *= m["best"]["odd"]
                tot_prob *= m["best"]["prob"] / 100.0
                grades.append(m["best"].get("grade", "C"))

            if tot_prob < 0.05:
                continue

            grade_score = sum(GRADE_W.get(g, 1) for g in grades)
            uniqueness  = len(set(m["best"]["name"] for m in combo)) / len(combo)
            score = tot_prob * math.log(max(tot_odd, 1.1)) * uniqueness * grade_score
            combos.append((score, combo, tot_odd, tot_prob, grades))

    if not combos:
        return jsonify({"error": "Не удалось собрать экспресс (слишком низкая вероятность)"})

    results = []
    for sc, combo, tot_odd, tot_prob, grades in sorted(combos, key=lambda x: x[0], reverse=True)[:3]:
        k_exp  = kelly_fraction(tot_prob, tot_odd) * (0.25 if len(combo) == 3 else 0.40)
        ev_exp = calc_ev(tot_prob, tot_odd)
        best_g = min(grades, key=lambda g: {"A": 0, "B": 1, "C": 2, "D": 3}.get(g, 3))
        results.append({
            "combo":      combo,
            "total_odd":  round(tot_odd,  2),
            "total_prob": round(tot_prob * 100, 2),
            "kelly":      round(k_exp * 100, 2),
            "ev":         round(ev_exp * 100, 1),
            "grade":      best_g,
        })

    return jsonify({"alternatives": results})


# ─── Football Today (ESPN unofficial API) ─────────────────────────────────────

_ESPN = 'https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard'
_MSK  = timezone(timedelta(hours=3))
_sched_cache = {'date': None, 'data': None, '_ts': 0, '_lock': threading.Lock()}
_H2H_CACHE   = {}   # (home_id, away_id) → (lh_factor, la_factor, count, ts)
_H2H_TTL     = 86400  # 24 hours

# ─── The Odds API — multi-market enrichment ───────────────────────────────────

ODDS_API_KEY  = os.environ.get('ODDS_API_KEY', '9f3e4a8a382f0bfa86bd679465732e15')
_OAPI_BASE    = 'https://api.the-odds-api.com/v4/sports'
_OAPI_LEAGUES = [
    'soccer_epl', 'soccer_germany_bundesliga', 'soccer_spain_la_liga',
    'soccer_italy_serie_a', 'soccer_france_ligue_one',
    'soccer_uefa_champs_league', 'soccer_uefa_europa_league',
    'soccer_conmebol_copa_libertadores',
]
_oapi_cache = {'date': None, 'data': {}, '_ts': 0, '_lock': threading.Lock()}


def _norm_name(name):
    n = unicodedata.normalize('NFKD', (name or '').lower())
    return ''.join(c for c in n if not unicodedata.combining(c)).strip()


def _avg_odd(prices):
    """Harmonic-mean average of decimal odds (averages implied probabilities)."""
    if not prices:
        return None
    return round(len(prices) / sum(1.0 / p for p in prices), 2)


def _fetch_oapi_today():
    """Fetch 1x2, totals (2.5 + 3.5), and BTTS odds from The Odds API.
    Results are cached per-day with a 4-hour TTL to conserve API quota.
    """
    if not ODDS_API_KEY:
        return {}
    now_ts = datetime.now().timestamp()
    today  = datetime.now(_MSK).date().isoformat()
    with _oapi_cache['_lock']:
        if (_oapi_cache.get('date') == today and _oapi_cache.get('data')
                and now_ts - _oapi_cache.get('_ts', 0) < 14400):
            return _oapi_cache['data']
    result = {}
    for league in _OAPI_LEAGUES:
        try:
            r = _req.get(
                f'{_OAPI_BASE}/{league}/odds/',
                params={'apiKey': ODDS_API_KEY, 'regions': 'eu',
                        'markets': 'h2h,totals,btts', 'oddsFormat': 'decimal'},
                timeout=10
            )
            if r.status_code != 200:
                continue
            for match in r.json():
                home_name = match.get('home_team', '')
                away_name = match.get('away_team', '')
                key = (_norm_name(home_name), _norm_name(away_name))
                h2h_h, h2h_d, h2h_a = [], [], []
                tb25, tm25, tb35, tm35 = [], [], [], []
                btts_y, btts_n = [], []
                for bm in match.get('bookmakers', []):
                    for mkt in bm.get('markets', []):
                        mk = mkt.get('key')
                        for oc in mkt.get('outcomes', []):
                            nm = oc.get('name', '')
                            p  = oc.get('price')
                            if not p:
                                continue
                            if mk == 'h2h':
                                if nm == home_name:   h2h_h.append(p)
                                elif nm == 'Draw':    h2h_d.append(p)
                                elif nm == away_name: h2h_a.append(p)
                            elif mk == 'totals':
                                pt = oc.get('point')
                                if pt == 2.5:
                                    (tb25 if nm == 'Over' else tm25).append(p)
                                elif pt == 3.5:
                                    (tb35 if nm == 'Over' else tm35).append(p)
                            elif mk == 'btts':
                                if nm == 'Yes': btts_y.append(p)
                                elif nm == 'No': btts_n.append(p)
                e = {}
                oh = _avg_odd(h2h_h); ox = _avg_odd(h2h_d); oa = _avg_odd(h2h_a)
                if oh: e['oh'] = oh
                if ox: e['ox'] = ox
                if oa: e['oa'] = oa
                if oh and ox and oa:
                    tot = 1/oh + 1/ox + 1/oa
                    p1 = (1/oh)/tot; px = (1/ox)/tot; p2 = (1/oa)/tot
                    e['o1x'] = round(1/(p1+px), 2)
                    e['ox2'] = round(1/(px+p2), 2)
                otb = _avg_odd(tb25); otm = _avg_odd(tm25)
                if otb:   e['otb']   = otb
                if otm:   e['otm']   = otm
                otb35 = _avg_odd(tb35); otm35 = _avg_odd(tm35)
                if otb35: e['otb35'] = otb35
                if otm35: e['otm35'] = otm35
                ob_y = _avg_odd(btts_y); ob_n = _avg_odd(btts_n)
                if ob_y: e['ob_yes'] = ob_y
                if ob_n: e['ob_no']  = ob_n
                if e:
                    result[key] = e
        except Exception:
            continue
    with _oapi_cache['_lock']:
        _oapi_cache['date'] = today
        _oapi_cache['_ts']  = now_ts
        _oapi_cache['data'] = result
    return result


def _form_pts(form_str):
    """Exponential-decay form score (0–15 scale, 7.5 = neutral).
    Most recent game carries full weight; each prior game decays by 0.80×.
    """
    s = (form_str or '').strip().upper()[-5:]
    if not s:
        return None
    decay  = [1.0, 0.80, 0.64, 0.51, 0.41]
    chars  = list(reversed(s))          # chars[0] = most recent game
    raw    = sum((3 if c == 'W' else 1.5 if c == 'D' else 0) * decay[i]
                 for i, c in enumerate(chars))
    max_w  = 3.0 * sum(decay[:len(chars)])
    return round(raw / max_w * len(chars) * 3, 1) if max_w else None


def _american_to_decimal(ml):
    try:
        ml = int(str(ml).replace('+', ''))
        if ml > 0:
            return round(ml / 100 + 1, 2)
        else:
            return round(100 / abs(ml) + 1, 2)
    except Exception:
        return None


def _g(d, *keys):
    """Safe nested get — treats None values same as missing keys."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _parse_espn_odds(comp):
    odds_list = comp.get('odds') or []
    if not odds_list:
        return {}
    od = odds_list[0]
    if not isinstance(od, dict):
        return {}
    ml  = od.get('moneyline') or {}
    tot = od.get('total') or {}
    h_ml = _g(ml, 'home', 'close', 'odds')
    x_ml = _g(ml, 'draw', 'close', 'odds') or _g(od, 'drawOdds', 'moneyLine')
    a_ml = _g(ml, 'away', 'close', 'odds')
    ov   = _g(tot, 'over',  'close', 'odds')
    un   = _g(tot, 'under', 'close', 'odds')
    result = {}
    oh = _american_to_decimal(h_ml) if h_ml else None
    ox = _american_to_decimal(x_ml) if x_ml else None
    oa = _american_to_decimal(a_ml) if a_ml else None
    if oh:  result['oh']  = oh
    if ox:  result['ox']  = ox
    if oa:  result['oa']  = oa
    if ov:  result['otb'] = _american_to_decimal(ov)
    if un:  result['otm'] = _american_to_decimal(un)
    # Derive double chance from 3-way moneyline (remove vig, combine probs)
    if oh and ox and oa:
        total = 1/oh + 1/ox + 1/oa
        p1 = (1/oh) / total
        px = (1/ox) / total
        p2 = (1/oa) / total
        result['o1x'] = round(1 / (p1 + px), 2)
        result['ox2'] = round(1 / (px + p2), 2)
    return result


@app.route('/football_today')
def football_today():
    msk_today = datetime.now(_MSK).date()
    today  = msk_today.isoformat()
    now_ts = datetime.now().timestamp()
    force = request.args.get('force') == '1'
    if not force:
        with _sched_cache['_lock']:
            if (_sched_cache.get('date') == today and _sched_cache.get('data')
                    and now_ts - _sched_cache.get('_ts', 0) < 3600):
                return jsonify(_sched_cache['data'])
    try:
        r = _espn_get(f'{_ESPN}?dates={today.replace("-","")}&limit=200', timeout=10)
        r.raise_for_status()
        resp = r.json()
        league_map = {str(lg.get('id', '')): (lg.get('name') or '') for lg in (resp.get('leagues') or [])}
        matches = []
        for ev in resp.get('events', []):
            try:
                comp = (ev.get('competitions') or [{}])[0]
                if not isinstance(comp, dict):
                    continue
                if _g(comp, 'status', 'type', 'state') == 'post':
                    continue
                cs = comp.get('competitors') or []
                if len(cs) < 2:
                    continue
                home = next((c for c in cs if c.get('homeAway') == 'home'), cs[0])
                away = next((c for c in cs if c.get('homeAway') == 'away'), cs[1])
                ts = ev.get('date', '')
                try:
                    dt_msk = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(_MSK)
                    if dt_msk.date() != msk_today:
                        continue
                    t = dt_msk.strftime('%H:%M')
                except Exception:
                    t = '—'
                uid = ev.get('uid', '')
                lg_id = uid.split('~l:')[1].split('~')[0] if '~l:' in uid else ''
                lg_name = league_map.get(lg_id) or ''
                if not lg_name:
                    notes = comp.get('notes') or []
                    lg_name = (notes[0].get('headline') or '') if notes else ''
                ev_state = _g(comp, 'status', 'type', 'state') or 'pre'
                ev_clock = _g(comp, 'status', 'type', 'shortDetail') if ev_state == 'in' else None
                matches.append({
                    'home':       (home.get('team') or {}).get('displayName', ''),
                    'away':       (away.get('team') or {}).get('displayName', ''),
                    'home_id':    (home.get('team') or {}).get('id', ''),
                    'away_id':    (away.get('team') or {}).get('id', ''),
                    'home_form':  home.get('form') or '',
                    'away_form':  away.get('form') or '',
                    'league':     lg_name,
                    'country':    '',
                    'time':       t,
                    'espn_id':    ev.get('id', ''),
                    'state':      ev_state,
                    'home_score': home.get('score') if ev_state == 'in' else None,
                    'away_score': away.get('score') if ev_state == 'in' else None,
                    'clock':      ev_clock,
                    'odds':       _parse_espn_odds(comp),
                })
            except Exception:
                continue
        oapi = _fetch_oapi_today()
        for m in matches:
            key = (_norm_name(m['home']), _norm_name(m['away']))
            if key in oapi:
                m['odds'].update(oapi[key])
        matches.sort(key=lambda x: x['time'])
        result = {'matches': matches, 'total': len(matches), 'date': today}
        with _sched_cache['_lock']:
            _sched_cache['date'] = today
            _sched_cache['_ts']  = now_ts
            _sched_cache['data'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/football_stats')
def football_stats():
    home_id   = request.args.get('home_id', '')
    away_id   = request.args.get('away_id', '')
    home_form = request.args.get('home_form', '')
    away_form = request.args.get('away_form', '')
    if not home_id or not away_id:
        return jsonify({'error': 'Missing team IDs'}), 400
    try:
        hs, hc, hg = get_team_season(home_id, 'home')
        as_, ac, ag = get_team_season(away_id, 'away')
        ng = max(hg, ag, 1)
        h2h_lh, h2h_la, h2h_n = get_h2h_factor(home_id, away_id)
        return jsonify({
            'hs': hs, 'hc': hc, 'as': as_, 'ac': ac,
            'fh': _form_pts(home_form), 'fa': _form_pts(away_form),
            'ng': ng,
            'h2h_lh': h2h_lh, 'h2h_la': h2h_la, 'h2h_n': h2h_n,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_all')
def analyze_all():
    today = datetime.now(_MSK).date().isoformat()
    with _sched_cache['_lock']:
        cached = _sched_cache.get('data')
    if not cached or cached.get('date') != today:
        return jsonify({'error': 'Сначала загрузите матчи (нажмите обновить)'}), 400

    to_analyze = [m for m in cached['matches']
                  if m.get('home_id') and m.get('away_id')
                  and m.get('odds', {}).get('oh')]
    if not to_analyze:
        return jsonify({'error': 'Нет матчей с коэффициентами'}), 400

    def analyze_one(m):
        try:
            hs, hc, hg = get_team_season(m['home_id'], 'home')
            as_, ac, ag = get_team_season(m['away_id'], 'away')
            ng = max(hg, ag, 1)
            fh = _form_pts(m.get('home_form', '')) or 7.5
            fa = _form_pts(m.get('away_form', '')) or 7.5
            h2h_lh, h2h_la, h2h_n = get_h2h_factor(m['home_id'], m['away_id'])
            result = _run_analysis(
                m['home'], m['away'], hs, hc, as_, ac, fh, fa, ng,
                m.get('odds', {}), m.get('league', ''), (h2h_lh, h2h_la)
            )
            result['h2h_n'] = h2h_n
            if not result.get('best'):
                return None
            result['time']    = m.get('time', '')
            result['league']  = m.get('league', '')
            result['espn_id'] = m.get('espn_id', '')
            result['match_time'] = m.get('time', '')
            result['_inputs'] = {
                'home_team': m['home'], 'away_team': m['away'],
                'hs': hs, 'hc': hc, 'as': as_, 'ac': ac,
                'fh': fh, 'fa': fa, 'ng': ng,
                'odds': m.get('odds', {}),
            }
            return result
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(analyze_one, to_analyze))

    results = [r for r in results if r]
    GRADE_O  = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    results.sort(key=lambda r: GRADE_O.get(r['best']['grade'], 3))
    SAVED_MATCHES.extend(results)
    return jsonify({'matches': results, 'total': len(results)})


@app.route('/check_results', methods=['POST'])
def check_results():
    event_ids = request.json.get('event_ids', [])
    out = {}
    for eid in event_ids:
        try:
            r = _espn_get(
                'https://site.api.espn.com/apis/site/v2/sports/soccer/all/summary',
                params={'event': eid}
            )
            if r.status_code != 200:
                continue
            comp = ((r.json().get('header') or {}).get('competitions') or [{}])[0]
            state = _g(comp, 'status', 'type', 'state')
            cs = comp.get('competitors') or []
            home = next((c for c in cs if c.get('homeAway') == 'home'), {})
            away = next((c for c in cs if c.get('homeAway') == 'away'), {})
            out[str(eid)] = {
                'state':      state or 'unknown',
                'home_score': home.get('score'),
                'away_score': away.get('score'),
                'clock':      _g(comp, 'status', 'type', 'shortDetail') if state == 'in' else None,
            }
        except Exception:
            continue
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True)
