from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.config import Config
from src.data_sources.market_snapshots import snapshot_rows_from_prediction, upsert_market_snapshot
from src.data_sources.prediction_snapshots import (
    load_prediction_snapshots,
    prediction_overview,
    prediction_snapshot_row,
    upsert_prediction_snapshot,
)
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
      --bg: #f3f5f1;
      --surface: #ffffff;
      --surface-2: #f8faf7;
      --ink: #17202a;
      --muted: #657386;
      --line: #d8ded7;
      --blue: #155eef;
      --blue-2: #0f49ba;
      --green: #147a55;
      --amber: #b45f06;
      --red: #b42318;
      --shadow: 0 12px 30px rgba(23, 32, 42, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    .shell {
      width: min(1320px, calc(100vw - 32px));
      margin: 0 auto;
    }
    header {
      background: #111820;
      color: white;
      border-bottom: 1px solid rgba(255,255,255,.12);
    }
    .topbar {
      min-height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    main { padding: 18px 0 34px; }
    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
    }
    .tab {
      height: 36px;
      padding: 0 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      font-weight: 760;
    }
    .tab.active {
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(280px, 1.4fr) 170px 118px 118px auto auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 14px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 760;
      text-transform: uppercase;
    }
    select, input, button {
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }
    select, input { width: 100%; padding: 0 11px; }
    button {
      padding: 0 15px;
      background: var(--blue);
      color: white;
      border-color: var(--blue);
      font-weight: 760;
      cursor: pointer;
      white-space: nowrap;
    }
    button:hover { background: var(--blue-2); }
    button.secondary {
      background: var(--surface);
      color: var(--ink);
      border-color: var(--line);
    }
    button:disabled { opacity: .55; cursor: wait; }
    .message {
      margin-bottom: 14px;
      padding: 11px 13px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--muted);
    }
    .error {
      border-color: #f4b5ae;
      background: #fff6f5;
      color: var(--red);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(340px, .9fr);
      gap: 14px;
      align-items: start;
    }
    .stack { display: grid; gap: 14px; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 15px;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    h2 {
      margin: 0;
      font-size: 14px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: var(--surface-2);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
    }
    .hero-score {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
      padding: 8px 0 14px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 14px;
    }
    .team-box {
      min-width: 0;
      display: grid;
      align-content: center;
      gap: 7px;
    }
    .team-name {
      font-size: clamp(23px, 3.2vw, 38px);
      font-weight: 850;
      line-height: 1.02;
      overflow-wrap: anywhere;
    }
    .score-box {
      min-width: 132px;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      justify-items: center;
      gap: 8px;
    }
    .goal {
      width: 54px;
      aspect-ratio: 1;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8f9fb;
      font-size: 30px;
      font-weight: 900;
      font-variant-numeric: tabular-nums;
    }
    .dash { color: var(--muted); font-weight: 850; }
    .prob-grid {
      display: grid;
      gap: 9px;
    }
    .prob {
      display: grid;
      grid-template-columns: minmax(116px, .5fr) minmax(120px, 1fr) 62px;
      gap: 10px;
      align-items: center;
    }
    .prob-label {
      min-width: 0;
      overflow-wrap: anywhere;
    }
    .bar {
      height: 11px;
      overflow: hidden;
      background: #e7ece8;
      border-radius: 999px;
    }
    .fill { height: 100%; background: var(--blue); }
    .fill.draw { background: var(--amber); }
    .fill.away { background: var(--green); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 9px;
      margin-top: 14px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      min-height: 68px;
      background: var(--surface-2);
    }
    .metric .k {
      color: var(--muted);
      font-size: 11px;
      font-weight: 740;
      text-transform: uppercase;
      margin-bottom: 5px;
    }
    .metric .v {
      font-size: 18px;
      font-weight: 850;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
    }
    .chart {
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }
    .chart-row {
      display: grid;
      grid-template-columns: 130px 1fr 58px;
      gap: 10px;
      align-items: center;
    }
    .chart-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
    }
    .chart-track {
      height: 16px;
      overflow: hidden;
      border-radius: 999px;
      background: #e7ece8;
    }
    .chart-fill {
      height: 100%;
      border-radius: 999px;
      background: var(--green);
    }
    .chart-fill.score { background: var(--blue); }
    .chart-fill.diff { background: var(--amber); }
    .chart-value {
      text-align: right;
      font-weight: 850;
      font-variant-numeric: tabular-nums;
    }
    .split {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    .list {
      display: grid;
      gap: 7px;
    }
    .row {
      display: grid;
      grid-template-columns: minmax(96px, .45fr) minmax(0, 1fr);
      gap: 10px;
      padding: 7px 0;
      border-bottom: 1px solid var(--line);
      align-items: baseline;
    }
    .row:last-child { border-bottom: 0; }
    .row .k {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
    }
    .row .v {
      min-width: 0;
      overflow-wrap: anywhere;
      font-variant-numeric: tabular-nums;
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
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      font-weight: 760;
      text-transform: uppercase;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #2c3440;
    }
    .empty {
      color: var(--muted);
      padding: 10px 0;
    }
    @media (max-width: 980px) {
      .toolbar, .layout, .split, .metric-grid { grid-template-columns: 1fr; }
      .hero-score { grid-template-columns: 1fr; }
      .score-box { justify-content: start; justify-items: start; grid-template-columns: auto auto auto; }
      .prob { grid-template-columns: 1fr; gap: 5px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="shell topbar">
      <h1>World Cup 2026 Predictor</h1>
    </div>
  </header>
  <main class="shell">
    <section class="tabs">
      <button id="predictTab" class="tab active">Predict</button>
      <button id="overviewTab" class="tab">Overview</button>
    </section>
    <section id="predictControls" class="toolbar">
      <label>Match
        <select id="matchSelect"></select>
      </label>
      <label>Weights
        <select id="weightPreset">
          <option value="0.3,0.7" selected>Live 30 / 70</option>
          <option value="0.4,0.6">Default 40 / 60</option>
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
      <button id="refreshButton" class="secondary">Reload</button>
    </section>
    <div id="status" class="message">Loading matches...</div>
    <section id="result" class="layout" style="display:none"></section>
    <section id="overview" class="stack" style="display:none"></section>
  </main>
  <script>
    const select = document.getElementById("matchSelect");
    const statusBox = document.getElementById("status");
    const result = document.getElementById("result");
    const predictButton = document.getElementById("predictButton");
    const refreshButton = document.getElementById("refreshButton");
    const predictTab = document.getElementById("predictTab");
    const overviewTab = document.getElementById("overviewTab");
    const predictControls = document.getElementById("predictControls");
    const overview = document.getElementById("overview");
    const weightPreset = document.getElementById("weightPreset");
    const modelWeight = document.getElementById("modelWeight");
    const marketWeight = document.getElementById("marketWeight");

    function pct(value) {
      const number = Number(value || 0);
      return `${(number * 100).toFixed(1)}%`;
    }
    function appliedWeight(lines, value) {
      return lines.length ? `each ${pct(value)}` : "not used";
    }
    function fixed(value, digits = 2) {
      const number = Number(value || 0);
      return number.toFixed(digits);
    }
    function whole(value) {
      const number = Number(value || 0);
      return Math.round(number).toLocaleString();
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
    function parseJsonish(value, fallback) {
      if (!value) return fallback;
      if (typeof value !== "string") return value;
      try { return JSON.parse(value); } catch { return fallback; }
    }
    function metric(label, value) {
      return `<div class="metric"><div class="k">${esc(label)}</div><div class="v">${esc(value)}</div></div>`;
    }
    function row(label, value) {
      return `<div class="row"><div class="k">${esc(label)}</div><div class="v">${esc(value)}</div></div>`;
    }
    function pill(value) {
      return `<span class="pill">${esc(value)}</span>`;
    }
    function probRow(label, value, cls = "") {
      const width = Math.max(0, Math.min(100, Number(value || 0) * 100));
      return `<div class="prob"><div class="prob-label">${esc(label)}</div><div class="bar"><div class="fill ${cls}" style="width:${width}%"></div></div><strong>${pct(value)}</strong></div>`;
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
      setStatus(`${payload.matches.length} matches loaded.`);
    }
    async function predict() {
      const matchId = select.value;
      if (!matchId) return;
      predictButton.disabled = true;
      setStatus(`Fetching market data and calculating ${matchId}...`);
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
        render(payload.prediction);
        setStatus("");
      } catch (error) {
        setStatus(error.message, true);
      } finally {
        predictButton.disabled = false;
      }
    }
    function marketRows(normalized, raw) {
      if (!Object.keys(normalized).length) {
        return `<tr><td colspan="3" class="empty">No accepted moneyline market</td></tr>`;
      }
      return Object.entries(normalized)
        .map(([key, value]) => `<tr><td>${esc(key)}</td><td>${pct(value)}</td><td>${raw[key] == null ? "" : fixed(raw[key], 3)}</td></tr>`)
        .join("");
    }
    function topScoreRows(top) {
      if (!top.length) return `<tr><td colspan="3" class="empty">No scorelines</td></tr>`;
      return top.map((item, idx) => `<tr><td>${idx + 1}</td><td>${esc(item.score)}</td><td>${pct(item.probability)}</td></tr>`).join("");
    }
    function totalLineRows(lines) {
      if (!lines.length) return `<tr><td colspan="4" class="empty">No full-match totals used</td></tr>`;
      return lines.map(item => `<tr><td>O/U ${esc(item.line)}</td><td>${pct(item.over_probability)}</td><td>${pct(item.over_price)}</td><td>${pct(item.under_price)}</td></tr>`).join("");
    }
    function teamTotalLineRows(lines) {
      if (!lines.length) return `<tr><td colspan="5" class="empty">No team totals used</td></tr>`;
      return lines.map(item => `<tr><td>${esc(item.team)}</td><td>O/U ${esc(item.line)}</td><td>${pct(item.over_probability)}</td><td>${pct(item.over_price)}</td><td>${pct(item.under_price)}</td></tr>`).join("");
    }
    function spreadLineRows(lines) {
      if (!lines.length) return `<tr><td colspan="4" class="empty">No full-match spreads used</td></tr>`;
      return lines.map(item => `<tr><td>${esc(item.team)}</td><td>${Number(item.line).toFixed(1)}</td><td>${pct(item.cover_probability)}</td><td>${pct(item.price)}</td></tr>`).join("");
    }
    function overviewRows(rows) {
      if (!rows.length) return `<tr><td colspan="10" class="empty">No saved predictions yet</td></tr>`;
      return rows.map(row => `<tr>
        <td>${esc(row.match_id)}</td>
        <td>${esc(row.home_team)} vs ${esc(row.away_team)}</td>
        <td>${esc(row.stage)}</td>
        <td>${esc(row.status)}</td>
        <td>${esc(row.prediction)}</td>
        <td>${esc(row.actual || "-")}</td>
        <td>${row.actual ? (row.correct_score ? "Yes" : "No") : "-"}</td>
        <td>${row.actual ? (row.correct_winner ? "Yes" : "No") : "-"}</td>
        <td>${row.actual ? (row.correct_goal_diff ? "Yes" : "No") : "-"}</td>
        <td>${esc(row.goal_error === "" ? "-" : row.goal_error)}</td>
      </tr>`).join("");
    }
    function hitRate(correct, total) {
      const denominator = Number(total || 0);
      if (!denominator) return 0;
      return Math.max(0, Math.min(1, Number(correct || 0) / denominator));
    }
    function chartRow(label, value, cls = "") {
      const width = Math.round(value * 100);
      return `<div class="chart-row">
        <div class="chart-label">${esc(label)}</div>
        <div class="chart-track"><div class="chart-fill ${cls}" style="width:${width}%"></div></div>
        <div class="chart-value">${width}%</div>
      </div>`;
    }
    async function loadOverview() {
      predictControls.style.display = "none";
      result.style.display = "none";
      overview.style.display = "grid";
      predictTab.classList.remove("active");
      overviewTab.classList.add("active");
      setStatus("Loading overview...");
      const response = await fetch("/api/overview");
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load overview");
      const summary = payload.summary || {};
      overview.innerHTML = `
        <section class="panel">
          <div class="panel-head"><h2>Prediction Overview</h2></div>
          <div class="metric-grid">
            ${metric("Saved predictions", whole(summary.predictions))}
            ${metric("Played predictions", whole(summary.played_predictions))}
            ${metric("Correct winners", `${whole(summary.correct_winners)} / ${whole(summary.played_predictions)}`)}
            ${metric("Correct scores", `${whole(summary.correct_scores)} / ${whole(summary.played_predictions)}`)}
          </div>
          <div class="metric-grid">
            ${metric("Correct goal diff", `${whole(summary.correct_goal_diffs)} / ${whole(summary.played_predictions)}`)}
            ${metric("Avg goal error", summary.average_goal_error === "" ? "-" : summary.average_goal_error)}
          </div>
          <div class="chart">
            ${chartRow("Winner", hitRate(summary.correct_winners, summary.played_predictions))}
            ${chartRow("Exact score", hitRate(summary.correct_scores, summary.played_predictions), "score")}
            ${chartRow("Goal diff", hitRate(summary.correct_goal_diffs, summary.played_predictions), "diff")}
          </div>
        </section>
        <section class="panel">
          <table>
            <thead><tr><th>ID</th><th>Match</th><th>Stage</th><th>Status</th><th>Prediction</th><th>Actual</th><th>Score</th><th>Winner</th><th>Goal diff</th><th>Goal err</th></tr></thead>
            <tbody>${overviewRows(payload.rows || [])}</tbody>
          </table>
        </section>`;
      setStatus("");
    }
    function showPredict() {
      predictControls.style.display = "grid";
      overview.style.display = "none";
      predictTab.classList.add("active");
      overviewTab.classList.remove("active");
      setStatus(`${select.options.length} matches loaded.`);
    }
    function render(pred) {
      const top = parseJsonish(pred.top_5_scorelines, []);
      const raw = parseJsonish(pred.moneyline_raw_prices, {});
      const normalized = parseJsonish(pred.moneyline_normalized_probabilities, {});
      const totalLines = parseJsonish(pred.total_lines_used, []);
      const teamTotalLines = parseJsonish(pred.team_total_lines_used, []);
      const spreadLines = parseJsonish(pred.spread_lines_used, []);
      const modelWeightText = `${pct(pred.model_weight)} model / ${pct(pred.moneyline_market_weight)} market`;
      result.innerHTML = `
        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <h2>Prediction</h2>
              <div class="pill-row">
                ${pill(pred.stage || "Stage n/a")}
              </div>
            </div>
            <div class="hero-score">
              <div class="team-box">
                <div class="team-name">${esc(pred.home_team)}</div>
              </div>
              <div class="score-box">
                <div class="goal">${esc(pred.predicted_home_goals)}</div>
                <div class="dash">-</div>
                <div class="goal">${esc(pred.predicted_away_goals)}</div>
              </div>
              <div class="team-box">
                <div class="team-name">${esc(pred.away_team)}</div>
              </div>
            </div>
            <div class="prob-grid">
              ${probRow(`${pred.home_team} win`, pred.home_win_prob)}
              ${probRow("Draw", pred.draw_prob, "draw")}
              ${probRow(`${pred.away_team} win`, pred.away_win_prob, "away")}
            </div>
            <div class="metric-grid">
              ${metric("Confidence", fixed(pred.confidence, 2))}
              ${metric("Blend", modelWeightText)}
              ${metric("Market match", fixed(pred.market_match_confidence, 2))}
              ${metric("WC games used", whole(pred.number_of_current_worldcup_matches_used))}
            </div>
          </section>
          <section class="panel">
            <div class="panel-head"><h2>Score Distribution</h2></div>
            <table>
              <thead><tr><th>#</th><th>Score</th><th>Probability</th></tr></thead>
              <tbody>${topScoreRows(top)}</tbody>
            </table>
          </section>
          <section class="panel">
            <div class="panel-head"><h2>Team Evidence</h2></div>
            <div class="split">
              <div class="list">
                ${row(pred.home_team, `${whole(pred.home_team_matches_used)} matches · ${fixed(pred.home_team_weighted_matches, 1)} recency-adjusted`)}
                ${row("Goals for", fixed(pred.home_team_goals_for))}
                ${row("Goals against", fixed(pred.home_team_goals_against))}
              </div>
              <div class="list">
                ${row(pred.away_team, `${whole(pred.away_team_matches_used)} matches · ${fixed(pred.away_team_weighted_matches, 1)} recency-adjusted`)}
                ${row("Goals for", fixed(pred.away_team_goals_for))}
                ${row("Goals against", fixed(pred.away_team_goals_against))}
              </div>
            </div>
            <div class="list" style="margin-top:12px">
              ${row("Head-to-head", `${whole(pred.head_to_head_matches_used)} matches · ${pred.head_to_head_home_wins}-${pred.head_to_head_draws}-${pred.head_to_head_away_wins}`)}
            </div>
          </section>
        </div>
        <aside class="stack">
          <section class="panel">
            <div class="panel-head"><h2>Polymarket</h2></div>
            <div class="list">
              ${row("Moneyline", pred.moneyline_used ? "Used" : "Not used")}
              ${row("Totals", pred.total_used ? pred.total_interpretation : "Not used")}
              ${row("Spreads", pred.spread_used ? pred.spread_interpretation : "Not used")}
              ${row("Source", pred.market_source || "none")}
            </div>
            <table style="margin-top:10px">
              <thead><tr><th>Outcome</th><th>Normalized</th><th>Raw</th></tr></thead>
              <tbody>${marketRows(normalized, raw)}</tbody>
            </table>
          </section>
          <section class="panel">
            <div class="panel-head"><h2>Totals Used</h2><span class="pill">${appliedWeight(totalLines, pred.total_per_line_weight)}</span></div>
            <table>
              <thead><tr><th>Line</th><th>Over prob</th><th>Over raw</th><th>Under raw</th></tr></thead>
              <tbody>${totalLineRows(totalLines)}</tbody>
            </table>
          </section>
          <section class="panel">
            <div class="panel-head"><h2>Team Totals Used</h2><span class="pill">${appliedWeight(teamTotalLines, pred.team_total_per_line_weight)}</span></div>
            <table>
              <thead><tr><th>Team</th><th>Line</th><th>Over prob</th><th>Over raw</th><th>Under raw</th></tr></thead>
              <tbody>${teamTotalLineRows(teamTotalLines)}</tbody>
            </table>
          </section>
          <section class="panel">
            <div class="panel-head"><h2>Spreads Used</h2><span class="pill">${appliedWeight(spreadLines, pred.spread_per_line_weight)}</span></div>
            <table>
              <thead><tr><th>Team</th><th>Line</th><th>Cover prob</th><th>Raw</th></tr></thead>
              <tbody>${spreadLineRows(spreadLines)}</tbody>
            </table>
          </section>
        </aside>`;
      result.style.display = "grid";
    }
    weightPreset.addEventListener("change", applyPreset);
    modelWeight.addEventListener("input", markCustom);
    marketWeight.addEventListener("input", markCustom);
    predictButton.addEventListener("click", predict);
    refreshButton.addEventListener("click", () => loadMatches().catch(error => setStatus(error.message, true)));
    predictTab.addEventListener("click", showPredict);
    overviewTab.addEventListener("click", () => loadOverview().catch(error => setStatus(error.message, true)));
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
    snapshot_rows = snapshot_rows_from_prediction(prediction)
    saved_rows = upsert_market_snapshot(cfg.market_snapshots_path, match_id, snapshot_rows)
    upsert_prediction_snapshot(cfg.prediction_snapshots_path, prediction_snapshot_row(prediction))
    return {"prediction": json_safe(prediction), "market_snapshot_rows_saved": saved_rows}


def overview_payload() -> dict[str, Any]:
    cfg = Config()
    state = build_state(cfg)
    snapshots = load_prediction_snapshots(cfg.prediction_snapshots_path)
    return json_safe(prediction_overview(state.schedule, snapshots))


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
            elif parsed.path == "/api/overview":
                self.send_json(overview_payload())
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
