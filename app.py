from flask import Flask, render_template, request, jsonify, send_from_directory
import math
from itertools import product, combinations
import uuid
import threading
from datetime import date, datetime, timezone, timedelta
import requests as _req

app = Flask(__name__)
SAVED_MATCHES = []

# ─── Constants ────────────────────────────────────────────────────────────────
AVG_HOME_GOALS        = 1.49   # Historical football home average
AVG_AWAY_GOALS        = 1.12   # Historical football away average
LEAGUE_GOALS_PER_GAME = 1.35   # League average goals scored per team per game
DC_RHO                = -0.13  # Dixon-Coles low-score correction factor
MAX_GOALS             = 7

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

def calculate_lambdas(hs, hc, as_, ac, ng=5):
    """
    Strength-index model using fixed league averages for normalization.

    Using LEAGUE_GOALS_PER_GAME as the baseline is more accurate than
    normalizing against the two teams' own average — that approach cancels
    out quality differences when both teams are equally strong or weak.

    ng: number of games the stats cover (default 5).
    """
    games = max(ng, 1)

    # Per-game rates
    hs_pg = hs / games
    hc_pg = hc / games
    as_pg = as_ / games
    ac_pg = ac / games

    L = LEAGUE_GOALS_PER_GAME

    # Strength relative to league average (>1 = above average)
    home_attack  = hs_pg / L   # how well home team scores
    away_defense = ac_pg / L   # how much away team concedes (>1 = leaky)
    away_attack  = as_pg / L
    home_defense = hc_pg / L   # how much home team concedes

    lh = home_attack * away_defense * AVG_HOME_GOALS
    la = away_attack * home_defense * AVG_AWAY_GOALS

    return max(min(lh, 4.5), 0.25), max(min(la, 4.5), 0.25)

def form_adjustment(lmbd, form_pts):
    """±20% max. form_pts 0–15 (5 games × 3 pts). 7.5 = neutral."""
    factor = 1.0 + (form_pts - 7.5) / 37.5
    return lmbd * max(factor, 0.50)

def home_away_bias(lh, la):
    return lh * 1.08, la * 0.93

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

def bet_grade(conf, ev_val):
    if conf >= 68 and ev_val >= 0.08: return "A"
    if conf >= 52 and ev_val >= 0.05: return "B"
    if conf >= 35 and ev_val >= 0.02: return "C"
    return "D"

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
        d = request.json
        home_team = (d.get("home_team") or "Хозяева").strip()
        away_team = (d.get("away_team") or "Гости").strip()

        hs  = max(int(d.get("hs",  0)), 0)
        hc  = max(int(d.get("hc",  0)), 0)
        as_ = max(int(d.get("as",  0)), 0)
        ac  = max(int(d.get("ac",  0)), 0)
        fh  = max(min(float(d.get("fh", 7.5)), 15.0), 0.0)
        fa  = max(min(float(d.get("fa", 7.5)), 15.0), 0.0)
        ng  = max(int(d.get("ng", 5)), 1)

        odds = d.get("odds", {})
        def o(key, default=2.0):
            val = odds.get(key)
            return max(float(val or default), 1.01)

        oh, ox, oa        = o("oh"), o("ox"), o("oa")
        o1x, ox2          = o("o1x", 1.5), o("ox2", 1.5)
        otb, otm          = o("otb", 1.8), o("otm", 1.8)
        otb35, otm35      = o("otb35", 2.1), o("otm35", 1.65)
        ob_yes, ob_no     = o("ob_yes", 1.7), o("ob_no", 1.9)

        lh, la = calculate_lambdas(hs, hc, as_, ac, ng)
        lh = form_adjustment(lh, fh)
        la = form_adjustment(la, fa)
        lh, la = home_away_bias(lh, la)

        probs, p1, px, p2 = match_probabilities(lh, la)
        vig = calc_vig(oh, ox, oa)
        def mp(odd): return fair_prob(odd, vig)

        markets = [
            ("П1",     p1,                                                      oh,     mp(oh)),
            ("X",      px,                                                      ox,     mp(ox)),
            ("П2",     p2,                                                      oa,     mp(oa)),
            ("1X",     p1 + px,                                                 o1x,    mp(o1x)),
            ("X2",     px + p2,                                                 ox2,    mp(ox2)),
            ("ТБ2.5",  sum(p for (h,a),p in probs.items() if h+a >= 3),        otb,    mp(otb)),
            ("ТМ2.5",  sum(p for (h,a),p in probs.items() if h+a <  3),        otm,    mp(otm)),
            ("ТБ3.5",  sum(p for (h,a),p in probs.items() if h+a >= 4),        otb35,  mp(otb35)),
            ("ТМ3.5",  sum(p for (h,a),p in probs.items() if h+a <  4),        otm35,  mp(otm35)),
            ("ОЗ Да",  sum(p for (h,a),p in probs.items() if h>0 and a>0),     ob_yes, mp(ob_yes)),
            ("ОЗ Нет", sum(p for (h,a),p in probs.items() if h==0 or  a==0),   ob_no,  mp(ob_no)),
        ]

        bets = {}
        for name, model_p, odd, market_p in markets:
            h_prob = hybrid_prob(model_p, market_p)
            ev_val = calc_ev(h_prob, odd)
            if ev_val <= 0:
                continue

            k_raw  = kelly_fraction(h_prob, odd)
            conf   = confidence_score(ev_val, model_p, market_p, k_raw, odd)
            k_dyn  = dynamic_kelly(k_raw, conf)
            grade  = bet_grade(conf, ev_val)
            vr     = model_p / max(market_p, 0.01)

            bets[name] = {
                "model_prob":  round(model_p  * 100, 1),
                "market_prob": round(market_p * 100, 1),
                "prob":        round(h_prob   * 100, 1),
                "odd":         round(odd, 2),
                "ev":          round(ev_val   * 100, 1),
                "kelly":       round(k_dyn    * 100, 2),
                "conf":        round(conf,     1),
                "grade":       grade,
                "vr":          round(vr, 2),
                "logical":     round(h_prob * (conf / 100), 4),
            }

        top3 = sorted(bets.items(), key=lambda x: x[1]["logical"], reverse=True)[:3]
        best = top3[0] if top3 else None
        best_pick = {"name": best[0], **best[1]} if best else None

        match_data = {
            "id":   str(uuid.uuid4()),
            "home": home_team,
            "away": away_team,
            "lh":   round(lh, 2),
            "la":   round(la, 2),
            "vig":  round(vig * 100, 2),
            "best": best_pick,
            "top3": top3,
        }
        SAVED_MATCHES.append(match_data)

        return jsonify({"top3": top3, "saved_matches": SAVED_MATCHES, "match": match_data})

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


# ─── Football Today (Sofascore) ───────────────────────────────────────────────

_SOFA = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Referer':    'https://www.sofascore.com/',
    'Accept':     'application/json, text/plain, */*',
    'Origin':     'https://www.sofascore.com',
}
_MSK  = timezone(timedelta(hours=3))
_sched_cache = {'date': None, 'data': None, '_lock': threading.Lock()}


@app.route('/football_today')
def football_today():
    today = date.today().isoformat()
    with _sched_cache['_lock']:
        if _sched_cache['date'] == today and _sched_cache['data']:
            return jsonify(_sched_cache['data'])
    try:
        r = _req.get(
            f'https://api.sofascore.com/api/v1/sport/football/scheduled-events/{today}',
            headers=_SOFA, timeout=10
        )
        r.raise_for_status()
        events = r.json().get('events', [])
        matches = []
        for ev in events:
            if ev.get('status', {}).get('type') not in ('notstarted', 'scheduled'):
                continue
            home = ev.get('homeTeam', {})
            away = ev.get('awayTeam', {})
            if not home.get('id') or not away.get('id'):
                continue
            ts = ev.get('startTimestamp', 0)
            tour = ev.get('tournament', {})
            matches.append({
                'home':    home.get('name', ''),
                'away':    away.get('name', ''),
                'home_id': home.get('id'),
                'away_id': away.get('id'),
                'league':  tour.get('name', ''),
                'country': tour.get('category', {}).get('name', ''),
                'time':    datetime.fromtimestamp(ts, tz=_MSK).strftime('%H:%M') if ts else '—',
            })
        matches.sort(key=lambda x: x['time'])
        result = {'matches': matches, 'total': len(matches), 'date': today}
        with _sched_cache['_lock']:
            _sched_cache['date'] = today
            _sched_cache['data'] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/football_stats')
def football_stats():
    home_id = request.args.get('home_id', type=int)
    away_id = request.args.get('away_id', type=int)
    if not home_id or not away_id:
        return jsonify({'error': 'Missing team IDs'}), 400

    def get_stats(team_id):
        try:
            r = _req.get(
                f'https://api.sofascore.com/api/v1/team/{team_id}/events/last/0',
                headers=_SOFA, timeout=8
            )
            evs = r.json().get('events', [])
            finished = [e for e in evs if e.get('status', {}).get('type') == 'finished'][-5:]
            scored = conceded = form_pts = 0
            for e in finished:
                is_home = (e.get('homeTeam') or {}).get('id') == team_id
                hs  = (e.get('homeScore') or {}).get('current', 0) or 0
                aws = (e.get('awayScore') or {}).get('current', 0) or 0
                if is_home:
                    scored += hs; conceded += aws
                    form_pts += 3 if hs > aws else (1 if hs == aws else 0)
                else:
                    scored += aws; conceded += hs
                    form_pts += 3 if aws > hs else (1 if aws == hs else 0)
            return scored, conceded, form_pts, max(len(finished), 1)
        except Exception:
            return 0, 0, 7, 5

    hs, hc, fh, hg = get_stats(home_id)
    as_, ac, fa, ag = get_stats(away_id)
    return jsonify({'hs': hs, 'hc': hc, 'as': as_, 'ac': ac,
                    'fh': fh, 'fa': fa, 'ng': max(hg, ag)})


if __name__ == "__main__":
    app.run(debug=True)
