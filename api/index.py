"""
WeatherMind — Vercel Serverless Backend
  Live data   : AccuWeather API  (current, 5-day forecast, hourly, alerts)
  Historical  : Open-Meteo Archive API (5 years daily — free, no key)
  ML model    : Trained fresh per-request using real historical data
                (Vercel is stateless — no in-memory persistence between calls)
  Frontend    : /public/index.html served by Vercel CDN
"""

import os, asyncio
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

AW_KEY  = os.getenv("ACCUWEATHER_API_KEY", "")
AW_BASE = "https://dataservice.accuweather.com"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
OM_ARCHIVE  = "https://archive-api.open-meteo.com/v1/archive"

app = FastAPI(title="WeatherMind API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FEATURES = [
    "day_of_year", "month", "season", "year",
    "precipitation_sum", "windspeed_max", "relative_humidity",
    "temp_lag1", "temp_lag7", "temp_roll14",
]

# ── WMO code helpers ──────────────────────────────────────────────────────────
def wmo_icon(code: int, is_day: bool = True) -> str:
    if code == 0:                         return "☀️" if is_day else "🌙"
    if code in (1, 2):                    return "🌤"
    if code == 3:                         return "☁️"
    if code in (45, 48):                  return "🌫"
    if code in (51, 53, 55):              return "🌦"
    if code in (61, 63, 65, 80, 81, 82):  return "🌧"
    if code in (71, 73, 75, 77, 85, 86):  return "🌨"
    if code in (95, 96, 99):              return "⛈"
    return "🌡"

def wmo_label(code: int) -> str:
    labels = {
        0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
        45:"Fog",48:"Icy fog",51:"Light drizzle",53:"Drizzle",55:"Heavy drizzle",
        61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",
        75:"Heavy snow",77:"Snow grains",80:"Light showers",81:"Showers",
        82:"Heavy showers",85:"Snow showers",86:"Heavy snow showers",
        95:"Thunderstorm",96:"Thunderstorm w/ hail",99:"Heavy thunderstorm",
    }
    return labels.get(code, f"Code {code}")

# ── HTTP helpers ──────────────────────────────────────────────────────────────
async def _get(url: str, params: dict) -> dict | list:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def _aw(path: str, params: dict = {}) -> dict | list:
    if not AW_KEY:
        raise HTTPException(503, "ACCUWEATHER_API_KEY is not set")
    return await _get(f"{AW_BASE}{path}", {**params, "apikey": AW_KEY})

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status":     "ok",
        "aw_key_set": bool(AW_KEY),
        "time":       datetime.utcnow().isoformat(),
    }

# ── City search (AccuWeather) ─────────────────────────────────────────────────
@app.get("/api/search")
async def search(q: str):
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty")
    data = await _aw("/locations/v1/cities/search", {"q": q, "language": "en-us"})
    if not data:
        raise HTTPException(404, f"No city found for '{q}'")
    return [
        {
            "key":     r["Key"],
            "name":    r["LocalizedName"],
            "state":   r.get("AdministrativeArea", {}).get("LocalizedName", ""),
            "country": r["Country"]["LocalizedName"],
            "lat":     r["GeoPosition"]["Latitude"],
            "lon":     r["GeoPosition"]["Longitude"],
            "elev":    round(r["GeoPosition"]["Elevation"]["Metric"]["Value"]),
        }
        for r in data[:8]
    ]

# ── Current conditions (AccuWeather) ─────────────────────────────────────────
@app.get("/api/current/{key}")
async def current(key: str):
    data = await _aw(f"/currentconditions/v1/{key}", {"details": "true"})
    d = data[0]
    return {
        "temp":       d["Temperature"]["Metric"]["Value"],
        "feels_like": d["RealFeelTemperature"]["Metric"]["Value"],
        "humidity":   d["RelativeHumidity"],
        "wind_speed": d["Wind"]["Speed"]["Metric"]["Value"],
        "wind_dir":   d["Wind"]["Direction"]["Localized"],
        "wind_gust":  d["WindGust"]["Speed"]["Metric"]["Value"],
        "uv":         d.get("UVIndex", 0),
        "uv_text":    d.get("UVIndexText", ""),
        "visibility": d["Visibility"]["Metric"]["Value"],
        "pressure":   d["Pressure"]["Metric"]["Value"],
        "dew_point":  d["DewPoint"]["Metric"]["Value"],
        "cloud_cover":d.get("CloudCover", 0),
        "condition":  d["WeatherText"],
        "icon":       d["WeatherIcon"],
        "is_day":     d["IsDayTime"],
        "precip_1h":  d.get("Precip1hr", {}).get("Metric", {}).get("Value", 0),
        "updated":    d["LocalObservationDateTime"],
    }

# ── 5-day forecast (AccuWeather) ──────────────────────────────────────────────
@app.get("/api/forecast/daily/{key}")
async def daily_forecast(key: str):
    data = await _aw(
        f"/forecasts/v1/daily/5day/{key}",
        {"details": "true", "metric": "true"},
    )
    return {
        "headline": data.get("Headline", {}).get("Text", ""),
        "days": [
            {
                "date":       d["Date"][:10],
                "hi":         d["Temperature"]["Maximum"]["Value"],
                "lo":         d["Temperature"]["Minimum"]["Value"],
                "feels_hi":   d["RealFeelTemperature"]["Maximum"]["Value"],
                "feels_lo":   d["RealFeelTemperature"]["Minimum"]["Value"],
                "day_icon":   d["Day"]["Icon"],
                "day_phrase": d["Day"]["LongPhrase"],
                "day_rain":   d["Day"]["PrecipitationProbability"],
                "day_wind":   d["Day"]["Wind"]["Speed"]["Value"],
                "day_wind_dir": d["Day"]["Wind"]["Direction"]["Localized"],
                "ngt_icon":   d["Night"]["Icon"],
                "ngt_phrase": d["Night"]["LongPhrase"],
                "ngt_rain":   d["Night"]["PrecipitationProbability"],
                "precip_mm":  d["Day"].get("TotalLiquid", {}).get("Value", 0),
                "sun_hours":  d.get("HoursOfSun", 0),
            }
            for d in data["DailyForecasts"]
        ],
    }

# ── 12-hour hourly forecast (AccuWeather) ─────────────────────────────────────
@app.get("/api/forecast/hourly/{key}")
async def hourly_forecast(key: str):
    data = await _aw(
        f"/forecasts/v1/hourly/12hour/{key}",
        {"details": "true", "metric": "true"},
    )
    return [
        {
            "time":     h["DateTime"][11:16],
            "temp":     h["Temperature"]["Value"],
            "feels":    h["RealFeelTemperature"]["Value"],
            "rain":     h["PrecipitationProbability"],
            "wind":     h["Wind"]["Speed"]["Value"],
            "wind_dir": h["Wind"]["Direction"]["Localized"],
            "humidity": h.get("RelativeHumidity", 0),
            "icon":     h["WeatherIcon"],
            "phrase":   h["IconPhrase"],
            "is_day":   h["IsDaylight"],
        }
        for h in data
    ]

# ── Active alerts (AccuWeather) ───────────────────────────────────────────────
@app.get("/api/alerts/{key}")
async def alerts(key: str):
    try:
        data = await _aw(f"/alerts/v1/{key}")
        return data or []
    except Exception:
        return []

# ── Open-Meteo: current + 7-day + 24h hourly (no key needed) ──────────────────
@app.get("/api/om/forecast")
async def om_forecast(lat: float, lon: float):
    params = {
        "latitude": lat, "longitude": lon,
        "current": (
            "temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
            "surface_pressure,is_day"
        ),
        "hourly": (
            "temperature_2m,precipitation_probability,precipitation,"
            "wind_speed_10m,relative_humidity_2m,weather_code"
        ),
        "daily": (
            "temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "precipitation_probability_max,wind_speed_10m_max,"
            "weather_code,sunrise,sunset"
        ),
        "forecast_days": 7,
        "timezone": "auto",
    }
    data = await _get(OM_FORECAST, params)
    c    = data["current"]
    code = c["weather_code"]
    is_day = bool(c.get("is_day", 1))

    now_str = c["time"][:13]
    hh = data["hourly"]
    hourly_out = []
    for i in range(len(hh["time"])):
        if hh["time"][i][:13] >= now_str:
            wc = hh.get("weather_code", [0]*len(hh["time"]))[i] if "weather_code" in hh else 0
            hourly_out.append({
                "time":    hh["time"][i][11:16],
                "temp":    hh["temperature_2m"][i],
                "rain":    hh["precipitation_probability"][i],
                "precip":  hh["precipitation"][i],
                "wind":    hh["wind_speed_10m"][i],
                "humidity":hh["relative_humidity_2m"][i],
                "icon":    wmo_icon(wc, hh["time"][i][11:13] not in ("20","21","22","23","00","01","02","03","04","05")),
            })
        if len(hourly_out) >= 24:
            break

    dd = data["daily"]
    daily_out = []
    for i in range(len(dd["time"])):
        wc = dd["weather_code"][i]
        daily_out.append({
            "date":     dd["time"][i],
            "hi":       dd["temperature_2m_max"][i],
            "lo":       dd["temperature_2m_min"][i],
            "precip":   dd["precipitation_sum"][i],
            "rain_pct": dd["precipitation_probability_max"][i],
            "wind":     dd["wind_speed_10m_max"][i],
            "code":     wc,
            "icon":     wmo_icon(wc),
            "phrase":   wmo_label(wc),
            "sunrise":  dd["sunrise"][i][11:16],
            "sunset":   dd["sunset"][i][11:16],
        })

    return {
        "current": {
            "temp":       c["temperature_2m"],
            "feels_like": c["apparent_temperature"],
            "humidity":   c["relative_humidity_2m"],
            "precip":     c["precipitation"],
            "wind_speed": c["wind_speed_10m"],
            "wind_dir":   c["wind_direction_10m"],
            "pressure":   c["surface_pressure"],
            "condition":  wmo_label(code),
            "icon":       wmo_icon(code, is_day),
            "is_day":     is_day,
            "code":       code,
        },
        "daily":   daily_out,
        "hourly":  hourly_out,
    }

# ── Open-Meteo: fetch 5-year historical data ──────────────────────────────────
async def _fetch_history(lat: float, lon: float) -> pd.DataFrame:
    end   = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily": (
            "temperature_2m_max,temperature_2m_min,precipitation_sum,"
            "wind_speed_10m_max,relative_humidity_2m_mean"
        ),
        "timezone": "UTC",
    }
    data = await _get(OM_ARCHIVE, params)
    dd   = data["daily"]
    df   = pd.DataFrame({
        "date":      pd.to_datetime(dd["time"]),
        "high_temp": dd["temperature_2m_max"],
        "low_temp":  dd["temperature_2m_min"],
        "precip":    dd["precipitation_sum"],
        "wind_max":  dd["wind_speed_10m_max"],
        "humidity":  dd["relative_humidity_2m_mean"],
    }).dropna()

    df["day_of_year"]        = df["date"].dt.dayofyear
    df["month"]              = df["date"].dt.month
    df["year"]               = df["date"].dt.year
    df["season"]             = ((df["month"] % 12) // 3).astype(int)
    df["precipitation_sum"]  = df["precip"]
    df["windspeed_max"]      = df["wind_max"]
    df["relative_humidity"]  = df["humidity"]
    df["temp_lag1"]          = df["high_temp"].shift(1)
    df["temp_lag7"]          = df["high_temp"].shift(7)
    df["temp_roll14"]        = df["high_temp"].rolling(14).mean()
    return df.dropna().reset_index(drop=True)

# ── ML training (called per-request on Vercel — stateless) ───────────────────
def _train(df: pd.DataFrame) -> dict:
    X  = df[FEATURES].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    yh = df["high_temp"].values
    yl = df["low_temp"].values

    Xtr, Xte, yh_tr, yh_te, yl_tr, yl_te = train_test_split(
        Xs, yh, yl, test_size=0.15, shuffle=False
    )
    params = dict(n_estimators=200, learning_rate=0.06, max_depth=5,
                  subsample=0.8, min_samples_leaf=5, random_state=42)

    mhi = GradientBoostingRegressor(**params).fit(Xtr, yh_tr)
    mlo = GradientBoostingRegressor(**params).fit(Xtr, yl_tr)

    yh_pred = mhi.predict(Xte)
    fi      = dict(zip(FEATURES, mhi.feature_importances_.tolist()))

    return dict(
        model_hi=mhi, model_lo=mlo, scaler=sc, df=df,
        metrics=dict(
            mae   = round(float(mean_absolute_error(yh_te, yh_pred)), 2),
            rmse  = round(float(np.sqrt(mean_squared_error(yh_te, yh_pred))), 2),
            r2    = round(float(r2_score(yh_te, yh_pred)), 3),
            n_train = len(Xtr),
            n_test  = len(Xte),
            feature_importance = {
                k: round(v, 4)
                for k, v in sorted(fi.items(), key=lambda x: -x[1])
            },
        ),
    )

# ── ML: yearly prediction endpoint ───────────────────────────────────────────
@app.get("/api/ml/yearly")
async def ml_yearly(lat: float, lon: float):
    """
    Fetches 5 years of real Open-Meteo data, trains the model, returns
    monthly predictions + metrics. Takes ~15-25s on first call per city.
    """
    loop = asyncio.get_event_loop()
    df   = await _fetch_history(lat, lon)
    cm   = await loop.run_in_executor(None, _train, df)

    MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    year   = datetime.utcnow().year
    preds  = []

    for mi in range(12):
        mid_doy = int(mi * 30.44 + 15)
        mdf = df[df["month"] == mi + 1]

        h_hi = round(float(mdf["high_temp"].mean()), 1) if len(mdf) else 0.0
        h_lo = round(float(mdf["low_temp"].mean()), 1)  if len(mdf) else 0.0
        h_pr = round(float(mdf["precip"].mean()), 1)    if len(mdf) else 0.0
        h_ws = round(float(mdf["wind_max"].mean()), 1)  if len(mdf) else 0.0
        h_hu = round(float(mdf["humidity"].mean()), 1)  if len(mdf) else 0.0
        lag  = h_hi

        X  = np.array([[mid_doy, mi+1, (mi % 12) // 3, year,
                         h_pr, h_ws, h_hu, lag, lag, lag]])
        Xs = cm["scaler"].transform(X)

        preds.append({
            "month":      MONTH_NAMES[mi],
            "pred_hi":    round(float(cm["model_hi"].predict(Xs)[0]), 1),
            "pred_lo":    round(float(cm["model_lo"].predict(Xs)[0]), 1),
            "hist_hi":    h_hi,
            "hist_lo":    h_lo,
            "hist_rain":  h_pr,
            "hist_wind":  h_ws,
            "hist_humid": h_hu,
        })

    return {
        "predictions": preds,
        "metrics":     cm["metrics"],
        "confidence":  round(cm["metrics"]["r2"] * 100, 1),
    }

# ── Combined: everything for a city in one call ───────────────────────────────
@app.get("/api/city/full")
async def city_full(key: str, lat: float, lon: float):
    """
    Parallel fetch: AccuWeather (current + 5-day + hourly + alerts)
    + Open-Meteo (7-day + 24h hourly). Does NOT block on ML training.
    Call /api/ml/yearly separately after this returns.
    """
    results = await asyncio.gather(
        current(key),
        daily_forecast(key),
        hourly_forecast(key),
        alerts(key),
        om_forecast(lat, lon),
        return_exceptions=True,
    )
    return {
        "aw_current": results[0] if not isinstance(results[0], Exception) else None,
        "aw_daily":   results[1] if not isinstance(results[1], Exception) else None,
        "aw_hourly":  results[2] if not isinstance(results[2], Exception) else None,
        "aw_alerts":  results[3] if not isinstance(results[3], Exception) else [],
        "om":         results[4] if not isinstance(results[4], Exception) else None,
    }
