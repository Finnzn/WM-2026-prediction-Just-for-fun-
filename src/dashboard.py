from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.config import Config
from src.models.predictor import build_state, predict_match_row, skip_reason


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>World Cup 2026 Predictor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #647386;
      --line: #d8dee8;
      --accent: #0969da;
      --accent-dark: #0757b8;
      --good: #0f8a5f;
      --warn: #a15c00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .wrap {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }
    .topbar {
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
    }
    main {
      padding: 22px 0 32px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) 190px 120px 120px auto auto;
      gap: 12px;
      align-items: end;
      margin-bottom: 18px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    select, input, button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
    }
    select, input {
      width: 100%;
      padding: 0 12px;
    }
    button {
      padding: 0 14px;
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      background: var(--panel);
      color: var(--ink);
      border-color: var(--line);
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.25fr .75fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 15px;
      letter-spacing: 0;
    }
    .score {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 10px 0 16px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 14px;
    }
    .team {
      flex: 1;
      min-width: 0;
    }
    .team .name {
      font-size: clamp(19px, 3vw, 30px);
      line-height: 1.1;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .goals {
      width: 64px;
      height: 58px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 28px;
      font-weight: 850;
      background: #f9fafb;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 9px;
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfe;
    }
    .prob {
      display: grid;
      grid-template-columns: 110px 1fr 58px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
    }
    .bar {
      height: 12px;
      background: #e9edf3;
      border-radius: 999px;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      background: var(--accent);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      min-height: 70px;
      background: #fbfcfe;
    }
    .stat .k {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .stat .v {
      font-size: 18px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .status-list {
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }
    .status-row {
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 10px;
      align-items: baseline;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--line);
    }
    .status-row .k {
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }
    .status-row .v {
      overflow-wrap: anywhere;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 4px;
      text-align: left;
    }
    th { color: var(--muted); font-weight: 750; }
    .message {
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--muted);
    }
    .error {
      border-color: #f0b8b8;
      color: #9a2b2b;
      background: #fff8f8;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 840px) {
      .controls, .grid, .stats { grid-template-columns: 1fr; }
      .score { align-items: stretch; }
      .goals { width: 52px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div>
        <h1>World Cup 2026 Predictor</h1>
        <div class="sub">Select a fixture, fetch live Polymarket prices, and blend them with the local model.</div>
      </div>
    </div>
  </header>
  <main class="wrap">
    <section class="controls">
      <label>Match
        <select id="matchSelect"></select>
      </label>
      <label>Weights
        <select id="weightPreset">
          <option value="0.3,0.7" selected>Live: 30 / 70</option>
          <option value="0.4,0.6">Default: 40 / 60</option>
          <option value="1,0">Model only</option>
          <option value="0,1">Market only</option>
          <option value="custom">Custom</option>
        </select>
      </label>
      <label>Model %
        <input id="modelWeight" type="number" min="0" max="100" step="5" value="30">
      </label>
      <label>Market %
        <input id="marketWeight" type="number" min="0" max="100" step="5" value="70">
      </label>
      <button id="predictButton">Predict</button>
      <button id="refreshButton" class="secondary">Reload Matches</button>
    </section>
    <div id="status" class="message">Loading matches...</div>
    <section id="result" class="grid" style="display:none"></section>
  </main>
  <script>
    const select = document.getElementById("matchSelect");
    const statusBox = document.getElementById("status");
    const result = document.getElementById("result");
    const predictButton = document.getElementById("predictButton");
    const refreshButton = document.getElementById("refreshButton");
    const weightPreset = document.getElementById("weightPreset");
    const modelWeight = document.getElementById("modelWeight");
    const marketWeight = document.getElementById("marketWeight");

    function pct(value) {
      const number = Number(value || 0);
      return `${(number * 100).toFixed(1)}%`;
    }
    function fixed(value, digits = 2) {
      const number = Number(value || 0);
      return number.toFixed(digits);
    }
    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[char]));
    }
    function setStatus(text, isError = false) {
      statusBox.textContent = text;
      statusBox.className = isError ? "message error" : "message";
      statusBox.style.display = text ? "block" : "none";
    }
    function probRow(label, value) {
      return `<div class="prob"><div>${esc(label)}</div><div class="bar"><div class="fill" style="width:${Math.max(0, Math.min(100, Number(value || 0) * 100))}%"></div></div><strong>${pct(value)}</strong></div>`;
    }
    function stat(label, value) {
      return `<div class="stat"><div class="k">${esc(label)}</div><div class="v">${esc(value)}</div></div>`;
    }
    function parseJsonish(value, fallback) {
      if (!value) return fallback;
      if (typeof value !== "string") return value;
      try { return JSON.parse(value); } catch { return fallback; }
    }
    function weights() {
      const model = Math.max(0, Number(modelWeight.value || 0)) / 100;
      const market = Math.max(0, Number(marketWeight.value || 0)) / 100;
      return { model, market };
    }
    function applyPreset() {
      if (weightPreset.value === "custom") return;
      const [model, market] = weightPreset.value.split(",").map(Number);
      modelWeight.value = Math.round(model * 100);
      marketWeight.value = Math.round(market * 100);
    }
    function markCustom() {
      const current = `${Number(modelWeight.value || 0) / 100},${Number(marketWeight.value || 0) / 100}`;
      const preset = Array.from(weightPreset.options).find(option => option.value === current);
      weightPreset.value = preset ? preset.value : "custom";
    }
    async function loadMatches() {
      result.style.display = "none";
      setStatus("Loading matches...");
      const response = await fetch("/api/matches");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load matches");
      select.innerHTML = "";
      payload.matches.forEach(match => {
        const option = document.createElement("option");
        option.value = match.match_id;
        option.textContent = `${match.match_id} · ${match.date} · ${match.home_team} vs ${match.away_team} · ${match.status}`;
        select.appendChild(option);
      });
      setStatus(`${payload.matches.length} matches loaded. Historical rows used by model: ${payload.state.number_of_historical_matches_used}.`);
    }
    async function predict() {
      const matchId = select.value;
      if (!matchId) return;
      predictButton.disabled = true;
      setStatus(`Fetching Polymarket and predicting ${matchId}...`);
      result.style.display = "none";
      try {
        const selectedWeights = weights();
        const params = new URLSearchParams({
          match_id: matchId,
          model_weight: String(selectedWeights.model),
          market_weight: String(selectedWeights.market)
        });
        const response = await fetch(`/api/predict?${params.toString()}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Prediction failed");
        render(payload);
        setStatus("");
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        predictButton.disabled = false;
      }
    }
    function render(payload) {
      const pred = payload.prediction;
      const top = parseJsonish(pred.top_5_scorelines, []);
      const raw = parseJsonish(pred.moneyline_raw_prices, {});
      const normalized = parseJsonish(pred.moneyline_normalized_probabilities, {});
      const topRows = top.map((row, idx) => `<tr><td>${idx + 1}</td><td>${esc(row.score)}</td><td>${pct(row.probability)}</td></tr>`).join("");
      const marketRows = Object.keys(normalized).length
        ? Object.entries(normalized).map(([key, value]) => `<tr><td>${esc(key)}</td><td>${pct(value)}</td><td>${raw[key] == null ? "" : fixed(raw[key], 3)}</td></tr>`).join("")
        : `<tr><td colspan="3">No accepted moneyline market</td></tr>`;
      result.innerHTML = `
        <section class="panel">
          <div class="meta">
            <span class="pill">${esc(pred.match_id)}</span>
            <span class="pill">${esc(pred.date)} ${esc(pred.kickoff_time)}</span>
            <span class="pill">${esc(pred.stage || "Stage n/a")}</span>
            <span class="pill">${esc(pred.status)}</span>
          </div>
          <div class="score">
            <div class="team"><div class="name">${esc(pred.home_team)}</div><div class="sub">expected ${fixed(pred.expected_home_goals)}</div></div>
            <div class="goals">${esc(pred.predicted_home_goals)}</div>
            <div class="goals">${esc(pred.predicted_away_goals)}</div>
            <div class="team"><div class="name">${esc(pred.away_team)}</div><div class="sub">expected ${fixed(pred.expected_away_goals)}</div></div>
          </div>
          <h2>Win / Draw / Loss</h2>
          ${probRow(`${pred.home_team} win`, pred.home_win_prob)}
          ${probRow("Draw", pred.draw_prob)}
          ${probRow(`${pred.away_team} win`, pred.away_win_prob)}
          <div class="stats" style="margin-top:14px">
            ${stat("Confidence", fixed(pred.confidence, 2))}
            ${stat("Weights", `${pct(pred.model_weight)} model / ${pct(pred.moneyline_market_weight)} market`)}
            ${stat("Historical matches", pred.number_of_historical_matches_used)}
            ${stat("Played WC 2026 matches used", pred.number_of_current_worldcup_matches_used)}
          </div>
        </section>
        <aside class="panel">
          <h2>Polymarket</h2>
          <div class="status-list">
            <div class="status-row"><div class="k">Moneyline</div><div class="v">${pred.moneyline_used ? "Used" : "Not used"}</div></div>
            <div class="status-row"><div class="k">Source</div><div class="v">${esc(pred.market_source || "none")}</div></div>
            <div class="status-row"><div class="k">Confidence</div><div class="v">${fixed(pred.market_match_confidence, 2)}</div></div>
          </div>
          <table>
            <thead><tr><th>Outcome</th><th>Normalized</th><th>Raw</th></tr></thead>
            <tbody>${marketRows}</tbody>
          </table>
          <h2 style="margin-top:18px">Top Scorelines</h2>
          <table>
            <thead><tr><th>#</th><th>Score</th><th>Probability</th></tr></thead>
            <tbody>${topRows}</tbody>
          </table>
          <h2 style="margin-top:18px">Notes</h2>
          <div class="mono">${esc(pred.notes || "none")}</div>
        </aside>`;
      result.style.display = "grid";
    }
    weightPreset.addEventListener("change", applyPreset);
    modelWeight.addEventListener("input", markCustom);
    marketWeight.addEventListener("input", markCustom);
    predictButton.addEventListener("click", predict);
    refreshButton.addEventListener("click", () => loadMatches().catch(error => setStatus(error.message, true)));
    loadMatches().catch(error => setStatus(error.message, true));
  </script>
</body>
</html>
"""


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def schedule_rows() -> dict[str, Any]:
    state = build_state()
    matches = []
    for row in state.schedule.itertuples(index=False):
        item = row._asdict()
        matches.append(
            {
                "match_id": item.get("match_id", ""),
                "date": item.get("date", ""),
                "kickoff_time": item.get("kickoff_time", ""),
                "home_team": item.get("home_team", ""),
                "away_team": item.get("away_team", ""),
                "status": item.get("status", ""),
                "stage": item.get("stage", ""),
                "group": item.get("group", ""),
            }
        )
    return {
        "matches": json_safe(matches),
        "state": {
            "number_of_historical_matches_used": state.number_of_historical_matches_used,
            "number_of_current_worldcup_matches_used": state.number_of_current_worldcup_matches_used,
            "training_data_start_date": state.training_data_start_date,
            "training_data_end_date": state.training_data_end_date,
        },
    }


def prediction_payload(match_id: str, model_weight: float | None = None, market_weight: float | None = None) -> dict[str, Any]:
    cfg = Config()
    state = build_state(cfg)
    rows = state.schedule[state.schedule["match_id"].eq(match_id)]
    if rows.empty:
        raise ValueError(f"No match found with match_id={match_id}")
    row = rows.iloc[0]
    reason = skip_reason(row)
    if reason:
        raise ValueError(f"{match_id} cannot be predicted: {reason}")
    prediction = predict_match_row(
        row,
        state,
        cfg,
        model_weight=model_weight,
        market_weight=market_weight,
        refresh_markets=True,
    )
    return {"prediction": json_safe(prediction)}


def parse_weight(value: str | None, name: str) -> float | None:
    if value in {None, ""}:
        return None
    try:
        weight = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if weight < 0:
        raise ValueError(f"{name} must be non-negative")
    return weight


class DashboardHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_html(INDEX_HTML)
            elif parsed.path == "/api/matches":
                self.send_json(schedule_rows())
            elif parsed.path == "/api/predict":
                params = parse_qs(parsed.query)
                match_id = (params.get("match_id") or [""])[0]
                if not match_id:
                    raise ValueError("Missing match_id")
                model_weight = parse_weight((params.get("model_weight") or [None])[0], "model_weight")
                market_weight = parse_weight((params.get("market_weight") or [None])[0], "market_weight")
                if model_weight is not None and market_weight is not None and model_weight + market_weight <= 0:
                    raise ValueError("At least one weight must be greater than zero")
                self.send_json(prediction_payload(match_id, model_weight=model_weight, market_weight=market_weight))
            else:
                self.send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local World Cup 2026 prediction dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
