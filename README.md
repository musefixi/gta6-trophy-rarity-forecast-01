# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/musefixi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/musefixi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the week-one earn rate of every Grand Theft Auto VI PS5 trophy on Exophase, scored by mean pinball loss on the logit scale. `main.py` (Python 3, standard library only, deterministic, no network) reads one case file per run — the house's timed reads of the game's trophy page up to a checkpoint — and writes five quantiles (5/25/50/75/95%) of each trophy's earn rate at the case's target.

**Method.** The reads are a step function: Exophase refreshes its counts on its own schedule, so the history is first reduced to its *distinct snapshots*, each weighted by how long it stayed current (capped at 3 hours, so one long-frozen page cannot dominate). On those snapshots each trophy's rate is fitted on the logit scale with a quadratic against ln(1 + hours since release), centred at the last snapshot — its level is the fitted rate there, its slope the current pace of progression, and its curvature how fast that pace bends as progression saturates. Slopes and curvatures are noisy per trophy, so each is shrunk toward the pooled value across all 55 trophies (partial pooling, weight growing with the trophy's snapshot count), and the extrapolated logit change over the horizon is softly capped. The spread combines a random-walk term growing with the square root of the horizon with the trophy's own fit residuals, shrunk toward the pooled residual scale — quiet trophies get tight bands, jumpy ones wide bands.

A history that never refreshed (one frozen page, players tracked flat — the launch-night practice cases) carries no trend at all: the event it is waiting for is the refresh itself, when a far larger tracked-player pool whose members have barely played dilutes every rate. That regime is forecast with a downward-shifted, wide logit distribution (a full 1.0-logit median drop, less for rates already at a clamp boundary, which can barely move past it) instead of a trend.

On synthetic final-like histories (saturating, noisy, and exactly-logit-linear ground truths), this forecaster's mean pinball loss beats both house baselines on the saturating and noisy truths and stays close on the smooth one; on the kit's practice cases it produces valid, deterministic forecasts whose bands are wide enough to survive the refresh the baselines' narrow bands missed.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with the case .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/<case id>.out for every *.in file found, in name order
# (with no env vars set it reads /input and writes /output, as the grader runs it)
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4yp70nzv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
