"""Forecast Exophase earn rates for every GTA VI PS5 trophy at a case's target.

Method
------
The house's reads are a step function: Exophase refreshes its counts on its
own schedule, so consecutive reads often repeat one page verbatim.  The
history is therefore first reduced to its distinct snapshots, each weighted
by how long it stayed current (capped, so one long-frozen page cannot
dominate).  On those snapshots each trophy's earn rate is modelled on the
logit scale as a quadratic against ln(1 + hours since release), centred
at the last snapshot: progression through the game is fast early and
bends as it saturates, and the logit keeps rates inside (0, 100) while
letting rare and common trophies share one scale -- the scale the pinball
loss is charged on.

Per-trophy slopes and curvatures are noisy, so each is shrunk toward the
pooled value across all trophies (partial pooling, weight growing with
the trophy's number of snapshots), and the extrapolated logit change over
the horizon is softly capped so a steep early climb cannot run away over
a 4.5-day horizon.  The
spread combines a random-walk term that grows with the square root of the
horizon (as the house baseline's does) with the trophy's own fit residuals,
shrunk toward the pooled residual scale, so quiet trophies get tight bands
and jumpy ones wide bands.

A history that never refreshed -- one frozen page, players tracked flat --
carries no trend at all, and the one event it is waiting for is the refresh
itself: a much larger tracked-player pool whose members have barely played
dilutes every rate.  That regime is forecast with a downward-shifted,
wide logit distribution instead of a trend.

Everything is deterministic: no randomness, no clock, no network.
"""

import json
import math
import os
from datetime import datetime, timezone
from statistics import NormalDist

_NORM = NormalDist()

# Logit clamp matching the scorer's: rates are clamped to [0.01, 99.99]
# percent before the logit is taken.
LO_RATE = 0.01
HI_RATE = 99.99

# Spread model constants.
RW_COEF = 0.52          # random-walk sigma per sqrt(day of horizon)
SIGMA_FLOOR = 0.35
SIGMA_CAP = 2.30
RESID_PRIOR_N = 6.0     # pseudo-count shrinking a trophy's residual scale
SLOPE_PRIOR_N = 5.0     # pseudo-count shrinking a trophy's slope
DUR_CAP_H = 3.0         # a snapshot's duration weight is capped at 3 hours
DELTA_SOFT = 2.2        # soft cap on extrapolated logit change
DELTA_HARD = 3.0

# Frozen-history (no refresh seen) regime.
FROZEN_SHIFT = 1.00     # median logit drop when the refresh lands
FROZEN_SIGMA = 1.45


def clamp_rate(r):
    return min(HI_RATE, max(LO_RATE, r))


def logit_pct(r):
    """Logit of a rate in percent, on the scorer's clamped scale."""
    p = clamp_rate(r) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    """Inverse logit, back to percent."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp to epoch seconds (float)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def release_epoch(case):
    """Epoch seconds of 00:00 UTC on the game's release date."""
    date = (case.get("game") or {}).get("release_date") or "2026-11-19"
    dt = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    return dt.timestamp()


def snapshots(case):
    """Reduce the read history to distinct snapshots with durations.

    Returns a list of (t_hours, players, rates, duration_hours) where
    t_hours is hours since release and duration_hours is how long the
    snapshot stayed the current page, ending at the case's checkpoint.
    """
    epoch = release_epoch(case)
    checkpoint = parse_time(case["at"])
    out = []
    for read in case.get("history") or []:
        t = parse_time(read.get("at") or read["slot"])
        key = (read.get("players_tracked"), tuple(
            None if r is None else round(float(r), 2)
            for r in (read.get("rates") or [])))
        if out and out[-1][3] == key:
            continue  # same page read again: not a new observation
        out.append([t, read.get("players_tracked"), read.get("rates") or [], key])
    snaps = []
    for i, (t, players, rates, _key) in enumerate(out):
        end = out[i + 1][0] if i + 1 < len(out) else checkpoint
        dur = max(0.05, min(DUR_CAP_H, (end - t) / 3600.0))
        snaps.append(((t - epoch) / 3600.0, players, rates, dur))
    return snaps


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b x; returns (a, b, resid_sd, x_bar, y_bar).

    resid_sd is the weighted residual standard deviation (0 with < 3 points).
    """
    sw = sum(ws)
    x_bar = sum(w * x for x, w in zip(xs, ws)) / sw
    y_bar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return y_bar, 0.0, 0.0, x_bar, y_bar
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = y_bar - b * x_bar
    if len(xs) >= 3:
        dof = max(1.0, sw - 2.0 * (sw / len(xs)))
        var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / dof
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return a, b, resid, x_bar, y_bar


def damp_delta(delta):
    """Softly cap an extrapolated logit change over the horizon."""
    if delta > DELTA_SOFT:
        return min(DELTA_HARD, DELTA_SOFT + 0.30 * (delta - DELTA_SOFT))
    if delta < -DELTA_SOFT:
        return max(-DELTA_HARD, -DELTA_SOFT + 0.30 * (delta + DELTA_SOFT))
    return delta


def forecast_case(case):
    """Return {trophy_id: [q05, q25, q50, q75, q95]} for one case."""
    trophies = case["trophies"]
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zetas = [_NORM.inv_cdf(tau) for tau in levels]
    epoch = release_epoch(case)
    t_target = (parse_time(case["target"]) - epoch) / 3600.0
    x_target = math.log1p(max(0.0, t_target))
    horizon_days = max(0.0, (case.get("horizon_minutes") or 0) / 1440.0)
    rw_sigma = RW_COEF * math.sqrt(horizon_days)

    snaps = snapshots(case)
    n_trophies = len(trophies)

    # Last observed rate per trophy (from the newest snapshot that has it).
    last_rate = [None] * n_trophies
    for _t, _n, rates, _d in snaps:
        for i in range(min(n_trophies, len(rates))):
            if rates[i] is not None:
                last_rate[i] = float(rates[i])

    players_first = next((n for _t, n, _r, _d in snaps if n), None)
    players_last = next((n for _t, n, _r, _d in reversed(snaps) if n), None)
    span_h = (snaps[-1][0] - snaps[0][0]) if len(snaps) >= 2 else 0.0
    grew = (players_first and players_last
            and players_last > players_first * 1.02)
    frozen = len(snaps) <= 2 or (not grew and span_h < 26.0)

    def row(mu, sigma):
        qs = [inv_logit_pct(mu + z * sigma) for z in zetas]
        qs = [round(min(100.0, max(0.0, q)), 4) for q in qs]
        for i in range(1, len(qs)):  # rounding must not break monotonicity
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        return qs

    if frozen:
        out = {}
        for i, tr in enumerate(trophies):
            r = last_rate[i] if last_rate[i] is not None else LO_RATE
            # A rate already at a clamp boundary can barely move past it:
            # a trophy everyone has earned is earned by the newcomers too,
            # so the refresh dilutes it far less than a mid-range rate.
            shift = FROZEN_SHIFT
            if r >= HI_RATE or r <= LO_RATE:
                shift *= 0.35
            out[tr["id"]] = row(logit_pct(r) - shift, FROZEN_SIGMA)
        return out

    # Per-trophy trend fits on the distinct snapshots.  With enough
    # snapshots the fit is a quadratic in x = ln(1 + hours), centred at
    # the last snapshot: its level is the fitted rate there, its slope
    # the current pace, and its curvature how fast that pace is bending
    # (progression saturates, so curvature is usually negative).
    def wls_quad(xs, ys, ws, xc):
        # Weighted least squares y ~ a + b (x - xc) + c (x - xc)^2.
        s = [0.0] * 5
        t = [0.0] * 3
        for x, y, w in zip(xs, ys, ws):
            u = x - xc
            xp = (1.0, u, u * u, u * u * u, u * u * u * u)
            for k in range(5):
                s[k] += w * xp[k]
            for k in range(3):
                t[k] += w * xp[k] * y
        m = [[s[0], s[1], s[2], t[0]],
             [s[1], s[2], s[3], t[1]],
             [s[2], s[3], s[4], t[2]]]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
            if abs(m[piv][col]) < 1e-12:
                return None
            m[col], m[piv] = m[piv], m[col]
            for r in range(3):
                if r != col:
                    f = m[r][col] / m[col][col]
                    for cc in range(4):
                        m[r][cc] -= f * m[col][cc]
        a, b, c = (m[i][3] / m[i][i] for i in range(3))
        sw = sum(ws)
        dof = max(1.0, sw - 3.0 * (sw / len(xs)))
        var = sum(w * (y - (a + b * (x - xc) + c * (x - xc) ** 2)) ** 2
                  for x, y, w in zip(xs, ys, ws)) / dof
        return a, b, c, math.sqrt(max(0.0, var))

    fits = []  # (level_at_last, slope_at_last, curvature, resid, n, x_last)
    for i in range(n_trophies):
        xs, ys, ws = [], [], []
        x_last = None
        for t, _n, rates, dur in snaps:
            if i < len(rates) and rates[i] is not None:
                x = math.log1p(max(0.0, t))
                xs.append(x)
                ys.append(logit_pct(float(rates[i])))
                ws.append(dur)
                x_last = x
        n = len(xs)
        if n >= 5:
            sol = wls_quad(xs, ys, ws, x_last)
            if sol is not None:
                fits.append((sol[0], sol[1], sol[2], sol[3], n, x_last))
                continue
        if n >= 2:
            a, b, resid, _xb, _yb = wls_line(xs, ys, ws)
            fits.append((a + b * x_last, b, 0.0, resid, n, x_last))
        elif n == 1:
            fits.append((ys[0], 0.0, 0.0, 0.0, n, x_last))
        else:
            fits.append((logit_pct(LO_RATE), 0.0, 0.0, 0.0, 0, None))

    # Pooled slope, curvature and residual scale (partial pooling).
    tot_w = sum(f[4] for f in fits)
    if tot_w > 0:
        b_pool = sum(f[1] * f[4] for f in fits) / tot_w
        c_pool = sum(f[2] * f[4] for f in fits) / tot_w
    else:
        b_pool = c_pool = 0.0
    resids = sorted(f[3] for f in fits if f[4] >= 3)
    s_pool = resids[len(resids) // 2] if resids else 0.25

    out = {}
    for i, tr in enumerate(trophies):
        level, b, curv, resid, n_pts, x_last = fits[i]
        if n_pts == 0 or x_last is None:
            r = last_rate[i] if last_rate[i] is not None else LO_RATE
            out[tr["id"]] = row(logit_pct(r), max(rw_sigma, SIGMA_FLOOR))
            continue
        w_slope = n_pts / (n_pts + SLOPE_PRIOR_N)
        b_shrunk = w_slope * b + (1.0 - w_slope) * b_pool
        c_shrunk = w_slope * curv + (1.0 - w_slope) * c_pool
        dx = x_target - x_last
        delta = damp_delta(b_shrunk * dx + c_shrunk * dx * dx)
        mu = level + delta
        if n_pts >= 3:
            s_i = math.sqrt((n_pts * resid ** 2 + RESID_PRIOR_N * s_pool ** 2)
                            / (n_pts + RESID_PRIOR_N))
        else:
            s_i = s_pool
        sigma = math.sqrt(rw_sigma ** 2 + s_i ** 2 * (1.0 + 2.0 / max(1, n_pts)))
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        out[tr["id"]] = row(mu, sigma)
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        forecast = forecast_case(case)
        rows = [{"id": tr["id"], "q": forecast[tr["id"]]}
                for tr in case["trophies"]]
        path = os.path.join(out_dir, case["id"] + ".out")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"trophies": rows}, fh)


if __name__ == "__main__":
    main()
