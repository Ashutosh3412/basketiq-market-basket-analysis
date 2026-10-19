from __future__ import annotations

from html import escape
from io import BytesIO
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# BasketIQ — Project-first Market Basket Analysis
# Upload is intentionally required; no local/default data is loaded.
# ============================================================

st.set_page_config(
    page_title="BasketIQ | Retail Pattern Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------ DESIGN SYSTEM ------------------------------

st.markdown(
r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  color-scheme: dark;
  --bg: #080D16;
  --bg2: #0D1726;
  --panel: #111E30;
  --panel-hi: #16263B;
  --line: #293D56;
  --text: #F4F7FC;
  --muted: #92A6BD;
  --soft: #C1CEDF;
  --lime: #C7F36B;
  --lime2: #E7FFA9;
  --blue: #7CA9FF;
  --cyan: #70E0CE;
  --purple: #B9A5FF;
}

html, body, [class*="css"], .stApp {
  font-family: 'Manrope', 'Segoe UI', sans-serif;
}
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse at 76% -3%, rgba(68,112,196,.17), transparent 34%),
    radial-gradient(ellipse at -8% 34%, rgba(199,243,107,.045), transparent 28%),
    var(--bg) !important;
  color: var(--text) !important;
}
/* Keep Streamlit chrome from covering the page's own brand bar. */
[data-testid="stHeader"] {
  background: #080D16 !important;
  height: 3.25rem !important;
  z-index: 1000 !important;
}
[data-testid="stToolbar"] {
  right: 1rem;
  background: #080D16 !important;
  visibility: hidden !important;
  pointer-events: none !important;
}
[data-testid="stDecoration"] { display: none !important; }
.block-container {
  max-width: 1480px;
  padding: 4.8rem 2.25rem 3.5rem 2.25rem !important;
}

h1,h2,h3,h4,h5,h6 { font-family:'Manrope','Segoe UI',sans-serif !important; color:var(--text) !important; letter-spacing:-.035em; }
p,li,label,.stMarkdown { color:var(--soft); }
[data-testid="stCaptionContainer"] p, small { color:var(--muted) !important; }
hr { border-color:var(--line) !important; }

/* Scroll progress line — supported in modern Chromium browsers. */
body::before {
  content:""; position:fixed; z-index:999999; left:0; top:0; height:3px; width:100%;
  transform:scaleX(0); transform-origin:left; pointer-events:none;
  background:linear-gradient(90deg,var(--lime),var(--cyan),var(--blue));
  animation:bqScrollProgress 1s linear both; animation-timeline:scroll(root);
}

/* Brand bar */
.productbar {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding:3px 2px 18px 2px; border-bottom:1px solid rgba(146,166,189,.17);
  margin-bottom:16px; animation:bqEnter .55s cubic-bezier(.16,1,.3,1) both;
}
.brand-lockup { display:flex; align-items:center; gap:12px; }
.brand-mark {
  width:39px;height:39px;display:flex;align-items:center;justify-content:center;
  border-radius:12px;background:var(--lime);color:#14200A;font-size:23px;font-weight:800;
  box-shadow:0 0 28px rgba(199,243,107,.17);
}
.brand-name { color:#F7F9FD !important;font-size:16px;font-weight:800;letter-spacing:-.04em;opacity:1 !important; }
.brand-name span { color:#C3D1E3 !important; }
.brand-sub { color:#B8C9DE !important;font:10px 'DM Mono',monospace;letter-spacing:.13em;text-transform:uppercase;margin-top:3px;opacity:1 !important; }
.top-status { display:flex;align-items:center;gap:8px;color:#C6D5E7 !important;font:10px 'DM Mono',monospace;letter-spacing:.08em;text-transform:uppercase;opacity:1 !important; }
.status-dot { width:7px;height:7px;border-radius:50%;background:var(--lime);box-shadow:0 0 11px rgba(199,243,107,.65); }

/* Large project-first landing hero */
.hero {
  position:relative;isolation:isolate;overflow:hidden;display:grid;
  grid-template-columns:minmax(0,1.32fr) minmax(250px,.68fr);gap:28px;align-items:center;
  padding:42px 42px 38px;margin:18px 0 26px;border:1px solid #2B405B;border-radius:24px;
  background:linear-gradient(118deg,#111D2F 0%,#142846 53%,#1A3551 100%);
  box-shadow:0 22px 65px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.035);
  animation:bqEnter .75s cubic-bezier(.16,1,.3,1) both;
}
.hero::after {
  content:"";position:absolute;z-index:0;top:-25%;bottom:-25%;left:-45%;width:26%;
  pointer-events:none;transform:skewX(-20deg);
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent);
  animation:bqSheen 1.5s .55s ease-out 1 both;
}
.hero-copy,.hero-art { position:relative;z-index:1; }
.eyebrow { color:var(--lime) !important;font:10px 'DM Mono',monospace;letter-spacing:.19em;text-transform:uppercase; }
.hero h1 { font-size:clamp(35px,4.2vw,60px) !important;line-height:1.02;margin:13px 0 15px !important;font-weight:800 !important;color:#FCFDFF !important; }
.hero h1 .highlight { color:var(--lime) !important; }
.hero p { max-width:690px;color:#C0D0E3 !important;font-size:15px;line-height:1.9;margin:0; }
.hero-tags { display:flex;flex-wrap:wrap;gap:8px;margin-top:21px; }
.hero-tag { border:1px solid rgba(173,195,222,.2);background:rgba(5,13,24,.28);padding:7px 10px;border-radius:8px;color:#D7E2EF !important;font:10px 'DM Mono',monospace;letter-spacing:.04em; }
.hero-cta { display:inline-flex;align-items:center;gap:9px;margin-top:25px;padding:11px 15px;border-radius:10px;background:var(--lime);color:#15200B !important;font-size:12px;font-weight:800;text-decoration:none !important;box-shadow:0 9px 24px rgba(199,243,107,.11);transition:transform .22s ease,box-shadow .22s ease; }
.hero-cta:hover { transform:translateY(-3px);box-shadow:0 14px 30px rgba(199,243,107,.2); }
.hero-art { min-height:235px;display:flex;align-items:center;justify-content:center; }
.orbit { position:absolute;width:205px;height:205px;border:1px solid rgba(181,201,226,.21);border-radius:50%;animation:bqOrbit 8s ease-in-out infinite; }
.orbit.two { width:154px;height:154px;border-style:dashed;border-color:rgba(199,243,107,.34);animation-delay:-2.4s; }
.orbit.three { width:100px;height:100px;border-color:rgba(124,169,255,.42);animation-delay:-4s; }
.orbit-core { width:56px;height:56px;border-radius:18px;background:var(--lime);color:#14200A;display:flex;align-items:center;justify-content:center;font-size:26px;font-weight:800;box-shadow:0 0 45px rgba(199,243,107,.23);transform:rotate(-8deg); }
.orbit-node { position:absolute;width:10px;height:10px;border-radius:50%;background:var(--lime);box-shadow:0 0 14px rgba(199,243,107,.6); }
.node1 { top:29px;right:calc(50% - 105px); }.node2 { bottom:34px;left:calc(50% - 79px);background:var(--blue); }.node3 { top:50%;right:calc(50% - 116px);background:var(--cyan); }
.art-label { position:absolute;bottom:0;right:0;color:#90A6C0;font:9px 'DM Mono',monospace;letter-spacing:.1em;text-transform:uppercase; }

/* Editorial section system */
.section-head { display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:29px 0 15px; }
.section-index { color:var(--lime) !important;font:10px 'DM Mono',monospace;letter-spacing:.15em;text-transform:uppercase; }
.section-title { color:#F3F7FD !important;font-size:clamp(21px,2.2vw,28px);font-weight:800;letter-spacing:-.045em;margin-top:5px;line-height:1.24; }
.section-desc { color:#93A7BF !important;font-size:12px;line-height:1.75;margin-top:5px;max-width:850px; }
.section-tail { border-bottom:1px solid #22344B;margin:18px 0 24px; }

/* Project explanation cards */
.info-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:16px 0 12px; }
.info-card { min-width:0;position:relative;overflow:hidden;background:linear-gradient(145deg,#111F32,#0D1726);border:1px solid #263A53;border-radius:16px;padding:21px;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;animation:bqEnter .7s cubic-bezier(.16,1,.3,1) both; }
.info-card:nth-child(2) { animation-delay:.10s; }.info-card:nth-child(3) { animation-delay:.20s; }
.info-card:hover { transform:translateY(-5px);border-color:#5C7899;box-shadow:0 17px 35px rgba(0,0,0,.22); }
.info-card::after { content:"";position:absolute;right:-24px;bottom:-35px;width:100px;height:100px;border:1px solid rgba(199,243,107,.10);border-radius:50%;box-shadow:0 0 0 18px rgba(124,169,255,.025),0 0 0 38px rgba(199,243,107,.018); }
.info-icon { display:flex;align-items:center;justify-content:center;width:38px;height:38px;border-radius:11px;margin-bottom:16px;background:rgba(199,243,107,.10);color:var(--lime);font-size:18px; }
.info-card:nth-child(2) .info-icon { background:rgba(124,169,255,.12);color:var(--blue); }
.info-card:nth-child(3) .info-icon { background:rgba(112,224,206,.11);color:var(--cyan); }
.info-card h3 { color:#F3F7FC !important;font-size:15px;margin:0 0 8px;letter-spacing:-.025em; }
.info-card p { color:#A6B8CC !important;font-size:12px;line-height:1.8;margin:0; }
.card-kicker { color:#8297B1 !important;font:9px 'DM Mono',monospace;letter-spacing:.12em;text-transform:uppercase;margin-bottom:8px; }

/* Pipeline */
.pipeline { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:17px 0; }
.pipeline-card { position:relative;min-width:0;padding:19px 17px;border-radius:14px;border:1px solid #263A52;background:#101B2B;transition:transform .22s ease,border-color .22s ease,background .22s ease;animation:bqEnter .62s cubic-bezier(.16,1,.3,1) both; }
.pipeline-card:nth-child(2){animation-delay:.08s}.pipeline-card:nth-child(3){animation-delay:.16s}.pipeline-card:nth-child(4){animation-delay:.24s}
.pipeline-card:hover { transform:translateY(-4px);border-color:#6585AA;background:#142239; }
.pipeline-no { color:var(--lime);font:11px 'DM Mono',monospace;letter-spacing:.08em; }
.pipeline-card h3 { margin:12px 0 7px !important;font-size:14px !important; }
.pipeline-card p { margin:0;color:#96AAC1 !important;font-size:11px;line-height:1.75; }
.pipeline-arrow { position:absolute;right:13px;top:15px;color:#54708E;font-size:15px; }

/* Metrics explained */
.metric-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:17px 0; }
.metric-explainer { padding:19px;border-radius:14px;background:linear-gradient(145deg,#111F32,#0D1726);border:1px solid #263A53;transition:transform .23s ease,border-color .23s ease; }
.metric-explainer:hover { transform:translateY(-4px);border-color:#5C7899; }
.metric-symbol { color:var(--lime);font:11px 'DM Mono',monospace;letter-spacing:.1em; }
.metric-explainer:nth-child(2) .metric-symbol{color:var(--blue)}.metric-explainer:nth-child(3) .metric-symbol{color:var(--cyan)}
.metric-explainer h3 { font-size:16px !important;margin:8px 0 6px !important; }
.metric-explainer p { color:#9FB1C6 !important;font-size:11px;line-height:1.8;margin:0; }
.metric-formula { display:inline-block;margin-top:12px;padding:6px 8px;border-radius:7px;background:#09121F;border:1px solid #253A54;color:#D8E4F3;font:10px 'DM Mono',monospace; }

/* Upload section */
.upload-shell { border:1px solid #324B68;border-radius:19px;padding:24px 24px 18px;background:linear-gradient(130deg,rgba(22,39,61,.95),rgba(13,23,38,.98));box-shadow:0 18px 46px rgba(0,0,0,.18); }
.upload-head { display:grid;grid-template-columns:52px 1fr;gap:15px;align-items:start; }
.upload-icon { width:50px;height:50px;display:flex;align-items:center;justify-content:center;border-radius:15px;background:var(--lime);color:#14200A;font-size:22px;font-weight:800;box-shadow:0 0 28px rgba(199,243,107,.12); }
.upload-head h3 { color:#F4F8FC !important;font-size:20px;margin:1px 0 6px !important; }
.upload-head p { color:#A7B9CF !important;font-size:12px;line-height:1.75;margin:0; }
.upload-specs { display:flex;flex-wrap:wrap;gap:8px;margin:15px 0; }
.upload-spec { border:1px solid #2C425C;background:#0C1726;color:#B9C9DB;padding:7px 9px;border-radius:8px;font:10px 'DM Mono',monospace; }
[data-testid="stFileUploaderDropzone"] { background:#0D192A !important;border:1px dashed #536D8E !important;border-radius:13px !important; }
[data-testid="stFileUploaderDropzone"] * { color:#DBE7F5 !important; }
[data-testid="stFileUploaderDropzone"] button { background:var(--lime) !important;border-color:var(--lime) !important;color:#14200A !important;font-weight:800 !important;border-radius:8px !important; }
[data-testid="stFileUploader"] small { color:#9CB0C8 !important; }

/* Result KPI cards */
.kpi-grid { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:15px 0 23px; }
.kpi { min-width:0;position:relative;overflow:hidden;padding:18px 19px 15px;border-radius:15px;background:linear-gradient(145deg,#142238,#0D1726);border:1px solid #2B405A;animation:bqEnter .62s cubic-bezier(.16,1,.3,1) both;transition:transform .24s ease,border-color .24s ease,box-shadow .24s ease; }
.kpi:nth-child(2){animation-delay:.08s}.kpi:nth-child(3){animation-delay:.16s}.kpi:nth-child(4){animation-delay:.24s}
.kpi::before { content:"";position:absolute;left:0;top:16px;bottom:16px;width:3px;border-radius:0 3px 3px 0;background:var(--accent,var(--lime)); }
.kpi::after { content:"";position:absolute;width:120px;height:120px;right:-64px;top:-64px;border-radius:50%;background:var(--accent,var(--lime));opacity:.045;transition:transform .3s ease; }
.kpi:hover { transform:translateY(-5px) scale(1.008);border-color:#5A789A;box-shadow:0 15px 32px rgba(0,0,0,.21); }
.kpi:hover::after { transform:scale(1.5); }
.kpi-head { display:flex;align-items:center;justify-content:space-between;gap:9px;color:#9CB0C7;font-size:10px;font-weight:700;letter-spacing:.04em; }
.kpi-icon { display:flex;align-items:center;justify-content:center;width:27px;height:27px;border-radius:8px;background:rgba(255,255,255,.055);color:var(--accent,var(--lime));font-size:14px; }
.kpi-value { color:#F7FAFD;font-size:clamp(24px,2.4vw,32px);font-weight:800;letter-spacing:-.055em;line-height:1.25;margin-top:11px;overflow-wrap:anywhere; }
.kpi-foot { color:#7F94AD;font:9px 'DM Mono',monospace;letter-spacing:.04em;margin-top:5px; }
.kpi-line { height:3px;border-radius:20px;background:#2A3C53;margin-top:12px;overflow:hidden; }
.kpi-line span { display:block;height:100%;width:var(--fill,70%);background:var(--accent,var(--lime));border-radius:20px;transform-origin:left;animation:bqLine 1s .2s both; }

/* Analysis sections / charts */
.result-section { margin:25px 0 27px; }
.panel { background:linear-gradient(145deg,#111E30,#0C1624);border:1px solid #293D56;border-radius:17px;padding:18px; }
.panel-heading { display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:13px; }
.panel-title { color:#ECF2F9 !important;font-size:13px;font-weight:800; }
.panel-tag { color:#DFFFA2 !important;background:rgba(199,243,107,.09);border:1px solid rgba(199,243,107,.22);border-radius:6px;padding:5px 8px;font:9px 'DM Mono',monospace;letter-spacing:.06em;white-space:nowrap; }
.panel-tag.blue { color:#B8D0FF !important;background:rgba(124,169,255,.09);border-color:rgba(124,169,255,.25); }
.insight { display:grid;grid-template-columns:36px minmax(0,1fr);gap:12px;align-items:start;border:1px solid rgba(199,243,107,.22);background:linear-gradient(100deg,rgba(199,243,107,.09),rgba(199,243,107,.025));border-radius:13px;padding:15px 17px;margin:13px 0 20px; }
.insight-icon { display:flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:10px;background:rgba(199,243,107,.12);color:var(--lime);font-size:16px; }
.insight-title { color:#EAF8D0 !important;font-size:12px;font-weight:800;margin-bottom:4px; }
.insight-text { color:#B5C5B1 !important;font-size:12px;line-height:1.75; }
.rule-grid { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:15px 0; }
.rule-card { min-width:0;padding:17px;border-radius:13px;background:#111F32;border:1px solid #2B405A;transition:transform .22s ease,border-color .22s ease,background .22s ease;animation:bqEnter .6s cubic-bezier(.16,1,.3,1) both; }
.rule-card:nth-child(2){animation-delay:.08s}.rule-card:nth-child(3){animation-delay:.16s}
.rule-card:hover { transform:translateY(-4px);border-color:#718BAA;background:#15263C;box-shadow:0 12px 28px rgba(0,0,0,.18); }
.rule-rank { color:var(--lime);font:10px 'DM Mono',monospace;letter-spacing:.1em; }
.rule-flow { display:grid;grid-template-columns:minmax(0,1fr) 20px minmax(0,1fr);gap:7px;align-items:center;margin:10px 0 13px; }
.rule-side { color:#F0F5FC;font-size:11px;font-weight:700;line-height:1.55;overflow-wrap:anywhere; }
.rule-arrow { color:var(--lime);font-size:18px;text-align:center; }
.rule-metrics { display:flex;gap:15px;flex-wrap:wrap;border-top:1px solid #293D55;padding-top:11px; }
.rule-metric-label { color:#8398B0;font:9px 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.06em; }
.rule-metric-value { color:#E9F0FA;font-size:12px;font-weight:800;margin-top:4px; }
.empty-state { padding:28px 20px;text-align:center;border:1px dashed #354B65;border-radius:14px;background:rgba(14,25,40,.5); }
.empty-state strong { display:block;color:#F0F5FB;margin-bottom:7px; }
.empty-state span { color:#9FB1C6;font-size:12px;line-height:1.7; }

/* Selectors and buttons */
div[data-baseweb="select"] > div { background:#121F31 !important;border-color:#31465F !important;color:#F3F7FC !important;border-radius:9px !important; }
div[data-baseweb="select"] * { color:#F3F7FC !important; }
input,textarea { background:#121F31 !important;color:#F3F7FC !important;border-color:#31465F !important; }
[data-testid="stSlider"] label,[data-testid="stSlider"] p { color:#DCE6F2 !important; }
.stButton button,.stDownloadButton button {
  background:var(--lime) !important;
  color:#10190A !important;
  -webkit-text-fill-color:#10190A !important;
  border:1px solid var(--lime) !important;
  border-radius:9px !important;
  font-weight:800 !important;
  opacity:1 !important;
  transition:transform .2s ease,box-shadow .2s ease,filter .2s ease;
}
.stButton button *, .stDownloadButton button *,
.stButton button p, .stDownloadButton button p,
.stButton button span, .stDownloadButton button span {
  color:#10190A !important;
  -webkit-text-fill-color:#10190A !important;
  opacity:1 !important;
  text-shadow:none !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
  color:#14200A !important;
  -webkit-text-fill-color:#14200A !important;
  opacity:1 !important;
}
.stButton button:hover,.stDownloadButton button:hover { transform:translateY(-2px);box-shadow:0 9px 23px rgba(199,243,107,.15);filter:brightness(1.04); }
.stButton button:active,.stDownloadButton button:active { transform:scale(.985); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid #293D56;border-radius:11px;overflow:hidden; }
[data-testid="stPlotlyChart"] { position:relative;border:1px solid #293D56;border-radius:15px;background:linear-gradient(145deg,#111E30,#0C1624);padding:6px 6px 0;transition:transform .24s ease,box-shadow .24s ease,border-color .24s ease;animation:bqEnter .55s cubic-bezier(.16,1,.3,1) both; }
[data-testid="stPlotlyChart"]:hover { z-index:3;transform:scale(1.006);border-color:#526E91;box-shadow:0 16px 38px rgba(0,0,0,.22); }
/* Enlarge the hovered bar itself, not just the chart container. */
.js-plotly-plot .barlayer .trace .points path,
.js-plotly-plot .barlayer .trace .point path {
  transform-box:fill-box !important;
  transform-origin:center center !important;
  transition:transform .22s cubic-bezier(.2,.75,.25,1),filter .22s ease,stroke .22s ease !important;
  cursor:pointer !important;
}
.js-plotly-plot .barlayer .trace .points path:hover,
.js-plotly-plot .barlayer .trace .point:hover path,
.js-plotly-plot .barlayer .trace .points .point:hover path {
  transform:scale(1.035,1.13) !important;
  filter:brightness(1.25) saturate(1.12) drop-shadow(0 0 5px rgba(199,243,107,.18)) !important;
  stroke:#F8FAFC !important;
  stroke-width:1.5px !important;
}
.js-plotly-plot .scatterlayer .trace .points path,
.js-plotly-plot .scatterlayer .trace .point path {
  transform-box:fill-box !important;
  transform-origin:center center !important;
  transition:transform .18s ease,filter .18s ease,stroke .18s ease !important;
  cursor:pointer !important;
}
.js-plotly-plot .scatterlayer .trace .points path:hover,
.js-plotly-plot .scatterlayer .trace .point:hover path {
  transform:scale(1.34) !important;
  filter:brightness(1.24) drop-shadow(0 0 6px rgba(124,169,255,.35)) !important;
  stroke:#F8FAFC !important;
  stroke-width:2px !important;
}
[data-testid="stExpander"] { border-color:#2B405A !important;border-radius:11px !important;background:rgba(16,29,46,.45); }
[data-testid="stExpander"] summary p { color:#E1EAF5 !important; }

/* Animated analysis waiting state */
.processing-shell {
  display:flex;align-items:center;gap:15px;padding:17px 19px;margin:12px 0 14px;
  border:1px solid #304763;border-radius:14px;
  background:linear-gradient(115deg,rgba(20,37,59,.96),rgba(12,23,38,.98));
}
.processing-orbit {
  width:38px;height:38px;flex:0 0 38px;border-radius:50%;position:relative;
  border:2px solid #314967;border-top-color:var(--lime);border-right-color:var(--cyan);
  animation:bqSpin .9s linear infinite;
}
.processing-orbit::after {
  content:"";position:absolute;left:10px;top:10px;width:14px;height:14px;border-radius:50%;
  background:var(--lime);box-shadow:0 0 18px rgba(199,243,107,.35);animation:bqPulse 1.1s ease-in-out infinite;
}
.processing-copy { min-width:0;flex:1; }
.processing-title { color:#F4F8FD;font-weight:800;font-size:13px;letter-spacing:-.01em; }
.processing-sub { margin-top:4px;color:#B7C7DB;font-size:11px;line-height:1.6; }
.processing-track { height:3px;background:#263B55;border-radius:20px;overflow:hidden;margin-top:10px; }
.processing-track span { display:block;width:38%;height:100%;border-radius:20px;background:linear-gradient(90deg,var(--lime),var(--cyan),var(--blue));animation:bqIndeterminate 1.2s ease-in-out infinite; }
@keyframes bqSpin { to { transform:rotate(360deg); } }
@keyframes bqPulse { 0%,100% { transform:scale(.78);opacity:.7; } 50% { transform:scale(1.12);opacity:1; } }
@keyframes bqIndeterminate { from { transform:translateX(-110%); } to { transform:translateX(290%); } }

/* Motion basics */
@keyframes bqEnter { from {opacity:0;transform:translateY(18px) scale(.99);filter:blur(2px)} to {opacity:1;transform:translateY(0) scale(1);filter:blur(0)} }
@keyframes bqSheen { 0%{left:-45%;opacity:0} 16%{opacity:.35} 100%{left:135%;opacity:0} }
@keyframes bqOrbit { 0%,100%{transform:scale(1);opacity:.85}50%{transform:scale(1.035);opacity:1} }
@keyframes bqLine { from{transform:scaleX(0)}to{transform:scaleX(1)} }
@keyframes bqScrollReveal { from {opacity:.15;transform:translateY(25px) scale(.99);filter:blur(2px)} to {opacity:1;transform:translateY(0) scale(1);filter:blur(0)} }
@keyframes bqScrollProgress { to {transform:scaleX(1)} }

/* Scroll-triggered reveals for modern Chromium; timed fallback above remains elsewhere. */
@supports (animation-timeline: view()) {
  .info-card, .pipeline-card, .metric-explainer, .kpi, .rule-card,
  [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], .result-section {
    animation-name:bqScrollReveal;
    animation-duration:1ms;
    animation-timing-function:linear;
    animation-fill-mode:both;
    animation-timeline:view();
    animation-range:entry 0% entry 55%;
  }
}
@media (prefers-reduced-motion: reduce) {
  *,*::before,*::after { animation-duration:.01ms !important;animation-iteration-count:1 !important;transition-duration:.01ms !important;scroll-behavior:auto !important; }
}
@media (max-width: 1050px) {
  .block-container { padding-left:1.25rem;padding-right:1.25rem; }
  .hero { grid-template-columns:minmax(0,1fr) 220px;padding:31px; }
  .info-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .pipeline { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (max-width: 700px) {
  .block-container { padding:4.25rem .85rem 2.5rem !important; }
  .top-status { display:none; }
  .hero { grid-template-columns:1fr;padding:25px 21px; }
  .hero-art { display:none; }
  .hero h1 { font-size:36px !important; }
  .info-grid,.pipeline,.metric-grid,.rule-grid { grid-template-columns:1fr; }
  .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr));gap:9px; }
  .kpi { padding:14px; }.kpi-value { font-size:24px; }
  .upload-shell { padding:17px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------ HELPERS ------------------------------

def safe_text(value) -> str:
    return escape(str(value), quote=True)


def itemset_text(itemset) -> str:
    return " + ".join(sorted(str(x) for x in itemset))


def section_head(index: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="section-head"><div><div class="section-index">{safe_text(index)}</div>'
        f'<div class="section-title">{safe_text(title)}</div>'
        f'<div class="section-desc">{safe_text(description)}</div></div></div>',
        unsafe_allow_html=True,
    )


def info_card(icon: str, kicker: str, title: str, body: str) -> str:
    return (
        '<div class="info-card">'
        f'<div class="info-icon">{safe_text(icon)}</div>'
        f'<div class="card-kicker">{safe_text(kicker)}</div>'
        f'<h3>{safe_text(title)}</h3><p>{safe_text(body)}</p>'
        '</div>'
    )


def kpi_card(label: str, value: str, icon: str, foot: str, accent: str, fill: str) -> str:
    return (
        f'<div class="kpi" style="--accent:{accent};--fill:{fill}">'
        f'<div class="kpi-head"><span>{safe_text(label)}</span><span class="kpi-icon">{safe_text(icon)}</span></div>'
        f'<div class="kpi-value">{safe_text(value)}</div>'
        f'<div class="kpi-foot">{safe_text(foot)}</div>'
        '<div class="kpi-line"><span></span></div></div>'
    )


def rule_card(rank: int, antecedent: str, consequent: str, support: float, confidence: float, lift: float) -> str:
    return (
        f'<div class="rule-card" style="--stagger:{min(rank,6) * 0.06:.2f}s">'
        f'<div class="rule-rank">SIGNAL / {rank:02d}</div>'
        f'<div class="rule-flow"><div class="rule-side">{safe_text(antecedent)}</div>'
        f'<div class="rule-arrow">→</div><div class="rule-side">{safe_text(consequent)}</div></div>'
        '<div class="rule-metrics">'
        f'<div><div class="rule-metric-label">Support</div><div class="rule-metric-value">{support:.2%}</div></div>'
        f'<div><div class="rule-metric-label">Confidence</div><div class="rule-metric-value">{confidence:.1%}</div></div>'
        f'<div><div class="rule-metric-label">Lift</div><div class="rule-metric-value">{lift:.2f}×</div></div>'
        '</div></div>'
    )


def empty_state(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-state"><strong>{safe_text(title)}</strong><span>{safe_text(body)}</span></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        # utf-8-sig handles common Excel-generated CSV files.
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(file_bytes), encoding="latin-1")
    return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")


def prepare_transactions(raw_data: pd.DataFrame, transaction_limit: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = raw_data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    col_map = {str(c).strip().lower(): c for c in data.columns}
    required = ["invoiceno", "description", "quantity"]
    missing = [c for c in required if c not in col_map]
    if missing:
        friendly = {"invoiceno": "InvoiceNo", "description": "Description", "quantity": "Quantity"}
        raise ValueError("Required columns missing: " + ", ".join(friendly.get(c, c) for c in missing))

    data = data.rename(columns={
        col_map["invoiceno"]: "InvoiceNo",
        col_map["description"]: "Description",
        col_map["quantity"]: "Quantity",
    })
    data = data.dropna(subset=["InvoiceNo", "Description"])
    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    data["Description"] = data["Description"].astype(str).str.strip()
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data = data.dropna(subset=["Quantity"])
    data = data[~data["InvoiceNo"].str.upper().str.startswith("C")]
    data = data[(data["Quantity"] > 0) & (data["Description"] != "")]
    if data.empty:
        raise ValueError("No valid purchase transactions remain after cleaning.")

    invoice_ids = data["InvoiceNo"].drop_duplicates().head(transaction_limit)
    data = data[data["InvoiceNo"].isin(invoice_ids)].copy()
    if data.empty:
        raise ValueError("No transactions remain in the selected sample.")

    basket = (
        data.groupby(["InvoiceNo", "Description"], observed=True)["Quantity"]
        .sum().unstack(fill_value=0).gt(0)
    )
    if basket.shape[1] < 2:
        raise ValueError("At least two distinct products are required for Apriori analysis.")
    return data, basket


def run_apriori(basket: pd.DataFrame, min_support: float, max_length: int) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    itemsets = apriori(basket, min_support=min_support, use_colnames=True, max_len=max_length)
    empty_rules = pd.DataFrame(columns=[
        "antecedents", "consequents", "antecedent support", "consequent support",
        "support", "confidence", "lift",
    ])
    if itemsets.empty or not itemsets["itemsets"].map(len).ge(2).any():
        return itemsets, empty_rules, True

    rules = association_rules(itemsets, metric="confidence", min_threshold=0.0)
    if rules.empty:
        return itemsets, rules, True

    expected_conf = rules["support"] / rules["antecedent support"]
    expected_lift = rules["confidence"] / rules["consequent support"]
    valid = bool(
        np.allclose(rules["confidence"].to_numpy(), expected_conf.to_numpy(), rtol=1e-6, atol=1e-8)
        and np.allclose(rules["lift"].to_numpy(), expected_lift.to_numpy(), rtol=1e-6, atol=1e-8)
    )
    return itemsets, rules, valid


def plotly_theme(fig, height: int = 430, left_margin: int = 45, right_margin: int = 35):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, Segoe UI, sans-serif", "size": 11, "color": "#C5D3E4"},
        title={"font": {"size": 15, "color": "#F4F7FC"}, "x": 0.01, "xanchor": "left"},
        height=height,
        margin={"l": left_margin, "r": right_margin, "t": 58, "b": 42},
        hoverlabel={"bgcolor": "#15263C", "font": {"color": "#FFFFFF", "family": "Manrope"}},
        transition={"duration": 500, "easing": "cubic-in-out"},
    )
    fig.update_xaxes(gridcolor="rgba(138,160,188,.13)", zerolinecolor="rgba(138,160,188,.18)", tickfont={"color": "#AFC0D2"}, title_font={"color": "#D5E1EF"})
    fig.update_yaxes(gridcolor="rgba(138,160,188,.07)", zerolinecolor="rgba(138,160,188,.18)", tickfont={"color": "#E0E9F4"}, title_font={"color": "#D5E1EF"})
    return fig


def render_hover_bar(fig, height: int, accent: str, chart_key: str) -> None:
    # Use a small HTML component so Plotly's native JavaScript hover events
    # can focus one bar, dim the others, and animate the hovered bar itself.
    import re

    chart_id = "bq-hover-" + re.sub(r"[^a-zA-Z0-9_-]", "-", chart_key)
    figure_json = fig.to_json().replace("</", "<" + "\\/")

    html_doc = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body { margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:transparent; }
  #__CHART_ID__ { width:100%; height:__HEIGHT__px; }
  .js-plotly-plot .barlayer path.point {
    transform-box:fill-box;
    transform-origin:center center;
    transition:transform 220ms cubic-bezier(.2,.75,.25,1), opacity 220ms ease,
      fill 220ms ease, filter 220ms ease, stroke 220ms ease;
    cursor:pointer;
  }
  @keyframes bqBarReveal {
    from { opacity:.15; transform:scaleX(.03); }
    to { opacity:1; transform:scaleX(1); }
  }
  .bq-chart-wrap { position:relative; width:100%; }
  .bq-hover-hint {
    position:absolute; right:12px; bottom:7px; z-index:3; pointer-events:none;
    font:9px Arial,sans-serif; color:#91A6BF; letter-spacing:.02em; opacity:.78;
  }
</style>
</head>
<body>
<div class="bq-chart-wrap">
  <div id="__CHART_ID__"></div>
  <div class="bq-hover-hint">HOVER A BAR TO FOCUS · MOVE AWAY TO RESET</div>
</div>
<script>
(function () {
  const chart = document.getElementById("__CHART_ID__");
  const figure = __FIGURE_JSON__;
  const highlightColor = "__ACCENT__";
  const mutedColor = "#263B53";
  const savedStyles = new WeakMap();
  let revealInstalled = false;

  function traceGroups() {
    return Array.from(chart.querySelectorAll(".barlayer .trace"));
  }
  function pathsFor(group) {
    if (!group) return [];
    let paths = Array.from(group.querySelectorAll("path.point"));
    if (!paths.length) paths = Array.from(group.querySelectorAll(".points path"));
    if (!paths.length) paths = Array.from(group.querySelectorAll("path"));
    return paths;
  }
  function captureBase(path) {
    if (savedStyles.has(path)) return;
    const cs = getComputedStyle(path);
    savedStyles.set(path, {
      fill: cs.fill,
      opacity: cs.opacity,
      filter: path.style.filter || "",
      transform: path.style.transform || "",
      stroke: cs.stroke,
      strokeWidth: cs.strokeWidth
    });
  }
  function allBars() {
    return traceGroups().flatMap(pathsFor);
  }
  function restoreBars() {
    allBars().forEach(function (path) {
      captureBase(path);
      const base = savedStyles.get(path);
      path.style.fill = base.fill;
      path.style.opacity = base.opacity;
      path.style.filter = base.filter;
      path.style.transform = "";
      path.style.stroke = base.stroke;
      path.style.strokeWidth = base.strokeWidth;
      path.style.animation = "none";
      path.style.transformOrigin = "center center";
    });
  }
  function installReveal() {
    if (revealInstalled) return;
    const bars = allBars();
    if (!bars.length) return;
    revealInstalled = true;
    bars.forEach(function (path, index) {
      captureBase(path);
      path.style.transformBox = "fill-box";
      path.style.transformOrigin = "left center";
      path.style.animation = "bqBarReveal 540ms cubic-bezier(.16,1,.3,1) " +
        Math.min(index * 32, 360) + "ms both";
      path.addEventListener("animationend", function () {
        path.style.animation = "none";
        path.style.transform = "";
        path.style.transformOrigin = "center center";
      }, {once:true});
    });
  }
  function highlightBar(eventData) {
    if (!eventData || !eventData.points || !eventData.points.length) return;
    const point = eventData.points[0];
    const groups = traceGroups();
    const group = groups[point.curveNumber] || groups[0];
    const bars = pathsFor(group);
    if (!bars.length) return;
    bars.forEach(function (path, index) {
      captureBase(path);
      path.style.animation = "none";
      path.style.transformOrigin = "center center";
      path.style.stroke = "transparent";
      path.style.strokeWidth = "0px";
      if (index === point.pointNumber) {
        path.style.fill = highlightColor;
        path.style.opacity = "1";
        path.style.filter = "brightness(1.10) drop-shadow(0 0 7px " + highlightColor + "55)";
        path.style.stroke = "#F8FAFC";
        path.style.strokeWidth = "1.5px";
        path.style.transform = "translateY(-1px) scale(1.025, 1.14)";
      } else {
        path.style.fill = mutedColor;
        path.style.opacity = ".22";
        path.style.filter = "saturate(.45) brightness(.72)";
        path.style.transform = "scale(1, .97)";
      }
    });
  }

  if (!window.Plotly) {
    document.body.innerHTML = '<div style="color:#F4F7FC;background:#0B1120;padding:20px;font:14px Arial">Chart library could not load. Check your internet connection and refresh BasketIQ.</div>';
    return;
  }
  Plotly.newPlot(chart, figure.data, figure.layout, {
    responsive: true,
    displaylogo: false,
    displayModeBar: true,
    scrollZoom: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]
  }).then(function () {
    installReveal();
    chart.on("plotly_hover", highlightBar);
    chart.on("plotly_unhover", restoreBars);
    chart.on("plotly_afterplot", function () {
      if (!revealInstalled) installReveal();
    });
  }).catch(function () {
    document.body.innerHTML = '<div style="color:#F4F7FC;background:#0B1120;padding:20px;font:14px Arial">Chart could not be rendered. Refresh BasketIQ and try again.</div>';
  });
})();
</script>
</body>
</html>'''

    html_doc = (
        html_doc
        .replace("__CHART_ID__", chart_id)
        .replace("__HEIGHT__", str(int(height)))
        .replace("__FIGURE_JSON__", figure_json)
        .replace("__ACCENT__", accent)
    )
    components.html(html_doc, height=int(height) + 18, scrolling=False)


def footer() -> None:
    st.markdown(
        '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border-top:1px solid #22344B;margin-top:37px;padding:17px 2px 0;color:#7489A2;font:9px DM Mono,monospace;letter-spacing:.07em;text-transform:uppercase">'
        '<span>BasketIQ / Retail pattern intelligence</span><span>Apriori · Association rules · Local analysis</span><span>Correlation is not causation</span></div>',
        unsafe_allow_html=True,
    )


# ------------------------------ BRAND / PROJECT INTRO ------------------------------

st.markdown(
    '<div class="productbar"><div class="brand-lockup"><div class="brand-mark">◈</div>'
    '<div><div class="brand-name">BasketIQ <span style="color:#91A5BD;font-weight:500">/ Retail Pattern Intelligence</span></div>'
    '<div class="brand-sub">Data mining project · Apriori algorithm</div></div></div>'
    '<div class="top-status"><span class="status-dot"></span> Project overview&nbsp;&nbsp;·&nbsp;&nbsp; upload-first workspace</div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="hero-copy">'
    '<div class="eyebrow">Retail intelligence / Market basket analysis</div>'
    '<h1>Find the pattern<br>behind <span class="highlight">the purchase.</span></h1>'
    '<p>BasketIQ turns transaction records into interpretable product relationships. Upload a transaction file, discover frequent itemsets, and explore association rules with support, confidence, and lift — without hiding the method behind a black box.</p>'
    '<div class="hero-tags"><span class="hero-tag">01 / DATA PREPARATION</span><span class="hero-tag">02 / APRIORI MINING</span><span class="hero-tag">03 / RULE INTERPRETATION</span></div>'
    '<a class="hero-cta" href="#dataset-upload">Start with your dataset <span aria-hidden="true">↓</span></a>'
    '</div><div class="hero-art"><div class="orbit"></div><div class="orbit two"></div><div class="orbit three"></div>'
    '<div class="orbit-node node1"></div><div class="orbit-node node2"></div><div class="orbit-node node3"></div>'
    '<div class="orbit-core">◈</div><div class="art-label">Transaction → Pattern → Signal</div></div></div>',
    unsafe_allow_html=True,
)

section_head("01 / Project overview", "What this project does", "A practical implementation of association-rule mining for retail transaction data.")
st.markdown(
    '<div class="info-grid">'
    + info_card("⌕", "THE PROBLEM", "Too many receipts, too little context", "A transaction file lists what was bought, but it does not immediately reveal which products repeatedly appear together across many orders.")
    + info_card("◈", "THE METHOD", "Discover combinations with Apriori", "Apriori searches for itemsets that meet a minimum support threshold, then generates directional rules and measures how useful each relationship may be.")
    + info_card("↗", "THE OUTPUT", "Readable signals, not a black box", "Explore ranked combinations, inspect support / confidence / lift, adjust thresholds, and export the results as CSV for a report or further analysis.")
    + '</div>',
    unsafe_allow_html=True,
)

section_head("02 / How it works", "From raw transactions to association rules", "The workflow is intentionally visible, so you can explain every stage during your project demonstration.")
pipeline_html = '<div class="pipeline">'
for number, title, desc, arrow in [
    ("01", "Prepare", "Remove cancelled invoices, missing product descriptions, and non-positive quantities.", "→"),
    ("02", "Encode", "Convert each invoice into a basket of products marked present or absent.", "→"),
    ("03", "Mine", "Use Apriori to keep itemsets meeting the configured minimum support.", "→"),
    ("04", "Interpret", "Generate rules and compare confidence and lift before drawing conclusions.", "✓"),
]:
    pipeline_html += (
        f'<div class="pipeline-card"><div class="pipeline-no">STAGE / {number}</div>'
        f'<div class="pipeline-arrow">{arrow}</div><h3>{safe_text(title)}</h3><p>{safe_text(desc)}</p></div>'
    )
pipeline_html += '</div>'
st.markdown(pipeline_html, unsafe_allow_html=True)

section_head("03 / Understand the metrics", "Three numbers tell different stories", "Use all three measures together: a high confidence alone is not enough to call a rule useful.")
st.markdown(
    '<div class="metric-grid">'
    '<div class="metric-explainer"><div class="metric-symbol">METRIC / 01</div><h3>Support</h3><p>How often the full itemset appears in all analyzed transactions. It helps distinguish common patterns from rare combinations.</p><div class="metric-formula">transactions containing the set ÷ all transactions</div></div>'
    '<div class="metric-explainer"><div class="metric-symbol">METRIC / 02</div><h3>Confidence</h3><p>Among transactions containing the left-hand side, how often does the right-hand side also appear? Direction matters.</p><div class="metric-formula">support(A ∪ B) ÷ support(A)</div></div>'
    '<div class="metric-explainer"><div class="metric-symbol">METRIC / 03</div><h3>Lift</h3><p>How much more often the consequent appears with the antecedent than its baseline frequency would suggest. Lift above 1 indicates positive association.</p><div class="metric-formula">confidence(A → B) ÷ support(B)</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div id="dataset-upload"></div>', unsafe_allow_html=True)
section_head("04 / Start here", "Bring your own transaction data", "BasketIQ does not load a bundled or local dataset automatically. Choose a CSV or Excel file to begin the analysis.")
st.markdown(
    '<div class="upload-shell"><div class="upload-head"><div class="upload-icon">↑</div><div>'
    '<h3>Upload a transaction dataset</h3>'
    '<p>Choose your own transaction export. The file is processed by this local app; no dataset is selected for you automatically.</p>'
    '<div class="upload-specs"><span class="upload-spec">CSV / XLSX / XLS</span><span class="upload-spec">REQUIRED: InvoiceNo</span><span class="upload-spec">Description</span><span class="upload-spec">Quantity</span></div>'
    '</div></div></div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose your file",
    type=["csv", "xlsx", "xls"],
    help="Required columns: InvoiceNo, Description, Quantity. Column capitalization does not matter.",
    key="basketiq_transaction_upload",
    label_visibility="visible",
)

if uploaded_file is None:
    st.markdown(
        '<div class="empty-state"><strong>Your analysis starts after upload</strong>'
        '<span>No file has been selected. Upload a CSV / XLSX / XLS above and BasketIQ will calculate the transaction metrics and charts below.</span></div>',
        unsafe_allow_html=True,
    )
    footer()
    st.stop()

# ------------------------------ UPLOAD & SETTINGS ------------------------------

st.markdown(
    '<div class="insight"><div class="insight-icon">✓</div><div>'
    '<div class="insight-title">File selected</div>'
    f'<div class="insight-text">{safe_text(uploaded_file.name)} is ready to be analyzed. Adjust the mining settings below and explore the results.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

section_head("05 / Configure analysis", "Tune the search", "These settings affect which patterns and association rules are shown. Raise support or confidence to make the criteria stricter.")
setting_cols = st.columns([1, 1, 1.1, 1.2], gap="medium")
with setting_cols[0]:
    min_support = st.slider("Minimum support", min_value=0.005, max_value=0.10, value=0.020, step=0.005, format="%.3f", help="Minimum fraction of analyzed transactions that must contain an itemset.")
with setting_cols[1]:
    min_confidence = st.slider("Minimum confidence", min_value=0.10, max_value=1.00, value=0.30, step=0.05, format="%.2f", help="Minimum conditional probability for an association rule.")
with setting_cols[2]:
    max_length = st.selectbox("Maximum items per set", [2, 3, 4], index=1, help="Larger itemsets increase computation and can be harder to interpret.")
with setting_cols[3]:
    transaction_limit = st.selectbox("Transactions to analyze", [500, 1000, 2000, 5000, 10000], index=0, help="Analyze up to this many distinct valid invoices, in their file order.")

try:
    with st.status("BasketIQ is analyzing your dataset", expanded=True) as analysis_status:
        st.markdown(
            '<div class="processing-shell">'
            '<div class="processing-orbit"></div>'
            '<div class="processing-copy">'
            '<div class="processing-title">Reading transactions and finding signals</div>'
            '<div class="processing-sub">BasketIQ is cleaning the file, encoding product baskets, mining itemsets, and checking rule metrics.</div>'
            '<div class="processing-track"><span></span></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        analysis_status.update(label="Step 1 of 3 · Loading transaction file", state="running", expanded=True)
        raw_data = load_dataset(uploaded_file.getvalue(), uploaded_file.name)
        analysis_status.update(label="Step 2 of 3 · Cleaning and encoding baskets", state="running", expanded=True)
        cleaned_data, basket = prepare_transactions(raw_data, transaction_limit)
        analysis_status.update(label="Step 3 of 3 · Mining itemsets and verifying rules", state="running", expanded=True)
        itemsets, all_rules, metrics_valid = run_apriori(basket, min_support, max_length)
        if all_rules.empty:
            filtered_rules = all_rules.copy()
        else:
            filtered_rules = all_rules[all_rules["confidence"] >= min_confidence].copy()
        analysis_status.update(
            label=f"Analysis complete · {len(basket):,} transactions processed",
            state="complete",
            expanded=False,
        )
except Exception as exc:
    st.error(f"Could not prepare the uploaded file: {exc}")
    st.info("Check that your dataset contains InvoiceNo, Description, and Quantity columns, with at least two distinct products.")
    footer()
    st.stop()

# ------------------------------ RESULTS ------------------------------

n_tx = len(basket)
n_products = basket.shape[1]
n_itemsets = len(itemsets)
n_rules = len(filtered_rules)
positive_rules = filtered_rules[filtered_rules["lift"] > 1].copy() if not filtered_rules.empty else filtered_rules.copy()
singletons = itemsets[itemsets["itemsets"].map(len).eq(1)].copy() if not itemsets.empty else itemsets.copy()
combos = itemsets[itemsets["itemsets"].map(len).ge(2)].copy() if not itemsets.empty else itemsets.copy()

section_head("06 / Analysis results", "Your transaction signals", "The cards below reflect the uploaded file and the active mining parameters.")
kpis = [
    ("Transactions", f"{n_tx:,}", "▦", "DISTINCT INVOICES IN SCOPE", "#C7F36B", "85%"),
    ("Unique products", f"{n_products:,}", "◈", "PRODUCT CATALOG WIDTH", "#7CA9FF", "68%"),
    ("Frequent itemsets", f"{n_itemsets:,}", "⌘", f"SUPPORT ≥ {min_support:.1%}", "#70E0CE", "76%"),
    ("Qualifying rules", f"{n_rules:,}", "↗", f"CONFIDENCE ≥ {min_confidence:.0%}", "#B9A5FF", "61%"),
]
st.markdown('<div class="kpi-grid">' + ''.join(kpi_card(*entry) for entry in kpis) + '</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="insight"><div class="insight-icon">✦</div><div>'
    '<div class="insight-title">Reading this run</div>'
    f'<div class="insight-text">BasketIQ analyzed <b>{n_tx:,}</b> invoices and <b>{n_products:,}</b> distinct products. It found <b>{n_itemsets:,}</b> itemsets meeting {min_support:.1%} support and <b>{n_rules:,}</b> rules meeting {min_confidence:.0%} confidence. Treat these as patterns in the analyzed sample, not proof of cause-and-effect.</div>'
    '</div></div>',
    unsafe_allow_html=True,
)

# ------------------------------ PRODUCT SIGNALS ------------------------------

section_head("07 / Product signals", "What appears most often", "Individual products and multi-product combinations are shown separately. Each chart uses a dedicated full-width panel to keep long labels readable.")

if itemsets.empty:
    empty_state("No frequent itemsets at this support threshold", "Lower minimum support or analyze a larger transaction sample.")
else:
    if not singletons.empty:
        prod = singletons.copy()
        prod["Product"] = prod["itemsets"].apply(lambda x: next(iter(x)))
        prod = prod.nlargest(12, "support").sort_values("support")
        fig_products = px.bar(
            prod,
            x="support",
            y="Product",
            orientation="h",
            text=prod["support"].map(lambda x: f"{x:.1%}"),
            color="support",
            color_continuous_scale=["#304968", "#4F85D7", "#C7F36B"],
            title="Most frequent individual products",
            custom_data=["Product"],
        )
        fig_products.update_traces(
            textposition="outside", cliponaxis=False, marker_line_width=0,
            hovertemplate="<b>%{customdata[0]}</b><br>Support: %{x:.2%}<extra></extra>",
        )
        fig_products.update_xaxes(tickformat=".0%", title="Share of transactions")
        fig_products.update_yaxes(title="", tickfont={"size": 10})
        fig_products.update_layout(showlegend=False, coloraxis_showscale=False)
        product_height = max(400, min(590, 120 + 32 * len(prod)))
        fig_products.update_layout(height=product_height)
        render_hover_bar(
            plotly_theme(fig_products, product_height, left_margin=250, right_margin=75),
            height=product_height,
            accent="#C7F36B",
            chart_key="individual-products",
        )

    if not combos.empty:
        combo = combos.copy()
        combo["CombinationFull"] = combo["itemsets"].apply(itemset_text)
        combo = combo.nlargest(10, "support").sort_values("support")
        combo["CombinationLabel"] = combo["CombinationFull"].apply(lambda x: x if len(x) <= 66 else x[:63] + "…")
        fig_combos = px.bar(
            combo,
            x="support",
            y="CombinationLabel",
            orientation="h",
            text=combo["support"].map(lambda x: f"{x:.1%}"),
            color="support",
            color_continuous_scale=["#2B4262", "#497BCC", "#70E0CE"],
            title="Frequent multi-product combinations",
            custom_data=["CombinationFull"],
        )
        fig_combos.update_traces(
            textposition="outside", cliponaxis=False, marker_line_width=0,
            hovertemplate="<b>%{customdata[0]}</b><br>Support: %{x:.2%}<extra></extra>",
        )
        fig_combos.update_xaxes(tickformat=".0%", title="Share of transactions")
        fig_combos.update_yaxes(title="", tickfont={"size": 10})
        fig_combos.update_layout(showlegend=False, coloraxis_showscale=False)
        combo_height = max(390, min(560, 115 + 36 * len(combo)))
        fig_combos.update_layout(height=combo_height)
        render_hover_bar(
            plotly_theme(fig_combos, combo_height, left_margin=360, right_margin=75),
            height=combo_height,
            accent="#70E0CE",
            chart_key="product-combinations",
        )

# ------------------------------ ASSOCIATION RULES ------------------------------

section_head("08 / Association intelligence", "Rules worth inspecting", "Ranked by lift, with support shown alongside it so rare high-lift rules aren't mistaken for broad patterns.")

if filtered_rules.empty:
    empty_state("No rules meet the current thresholds", "Lower minimum support or confidence, or choose a larger transaction sample.")
else:
    ranked = filtered_rules.sort_values(["lift", "support"], ascending=False).copy()
    top_rules = ranked.head(3)
    st.markdown('<div class="rule-grid">' + ''.join(
        rule_card(i, itemset_text(row["antecedents"]), itemset_text(row["consequents"]), float(row["support"]), float(row["confidence"]), float(row["lift"]))
        for i, (_, row) in enumerate(top_rules.iterrows(), start=1)
    ) + '</div>', unsafe_allow_html=True)

    best = ranked.iloc[0]
    if float(best["support"]) < max(min_support * 2, 0.05):
        st.markdown(
            '<div class="insight"><div class="insight-icon">!</div><div>'
            '<div class="insight-title">Interpretation guardrail</div>'
            f'<div class="insight-text">The highest-lift rule has {float(best["support"]):.2%} support. It may be worth inspecting, but it is rare in this sample. Lift signals association, not causation.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    scatter = ranked.head(900).copy()
    scatter["Rule"] = scatter.apply(lambda row: f"{itemset_text(row['antecedents'])} → {itemset_text(row['consequents'])}", axis=1)
    fig_rules = px.scatter(
        scatter,
        x="confidence",
        y="lift",
        size="support",
        color="lift",
        hover_name="Rule",
        hover_data={"support": ":.2%", "confidence": ":.1%", "lift": ":.2f"},
        color_continuous_scale=["#44628A", "#7CA9FF", "#C7F36B"],
        title="Rule landscape · confidence vs. lift",
        opacity=.86,
        size_max=20,
    )
    fig_rules.add_hline(y=1, line_dash="dot", line_color="#9BAEC2", annotation_text="Lift = 1 baseline", annotation_font_color="#C5D2E0")
    fig_rules.update_xaxes(title="Confidence", tickformat=".0%", range=[0, 1.04])
    fig_rules.update_yaxes(title="Lift", rangemode="tozero")
    fig_rules.update_layout(coloraxis_colorbar={"title": "Lift", "tickfont": {"color": "#C7D5E6"}})
    st.plotly_chart(plotly_theme(fig_rules, 470, left_margin=55, right_margin=65), width="stretch", config={"displaylogo": False})

    with st.expander("Inspect all qualifying rules", expanded=False):
        rules_table = ranked[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        rules_table["If customer buys"] = rules_table["antecedents"].apply(itemset_text)
        rules_table["Then may also buy"] = rules_table["consequents"].apply(itemset_text)
        rules_table = rules_table[["If customer buys", "Then may also buy", "support", "confidence", "lift"]]
        view_table = rules_table.rename(columns={"support": "Support", "confidence": "Confidence", "lift": "Lift"}).copy()
        view_table["Support"] = view_table["Support"].map(lambda x: f"{x:.2%}")
        view_table["Confidence"] = view_table["Confidence"].map(lambda x: f"{x:.1%}")
        view_table["Lift"] = view_table["Lift"].map(lambda x: f"{x:.2f}")
        st.dataframe(view_table, width="stretch", hide_index=True, height=430)

    export_rules = ranked[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    export_rules["If customer buys"] = export_rules["antecedents"].apply(itemset_text)
    export_rules["Then may also buy"] = export_rules["consequents"].apply(itemset_text)
    export_rules = export_rules[["If customer buys", "Then may also buy", "support", "confidence", "lift"]]
    st.download_button(
        "Export qualifying association rules · CSV",
        data=export_rules.to_csv(index=False).encode("utf-8-sig"),
        file_name="association_rules.csv",
        mime="text/csv",
        key="export_rules_final",
    )

# ------------------------------ ITEMSET TABLE / EXPORT ------------------------------

section_head("09 / Explore the patterns", "Search and export frequent itemsets", "Use the filters to narrow the table, then export the matching patterns.")
if itemsets.empty:
    empty_state("No itemsets to explore", "Change support or sample size to discover patterns.")
else:
    filter_cols = st.columns([1, 1.3], gap="medium")
    with filter_cols[0]:
        size_choice = st.selectbox("Itemset type", ["All sizes", "Single products", "Pairs", "3+ products"], key="final_itemset_size")
    with filter_cols[1]:
        search_term = st.text_input("Search product or combination", placeholder="Try: mug, hand warmer, lantern…", key="final_itemset_search")

    itemsets_table = itemsets.copy()
    itemsets_table["Products"] = itemsets_table["itemsets"].apply(itemset_text)
    itemsets_table["Items"] = itemsets_table["itemsets"].map(len)
    if size_choice == "Single products":
        itemsets_table = itemsets_table[itemsets_table["Items"] == 1]
    elif size_choice == "Pairs":
        itemsets_table = itemsets_table[itemsets_table["Items"] == 2]
    elif size_choice == "3+ products":
        itemsets_table = itemsets_table[itemsets_table["Items"] >= 3]
    if search_term.strip():
        itemsets_table = itemsets_table[itemsets_table["Products"].str.contains(search_term.strip(), case=False, na=False)]
    itemsets_table = itemsets_table.sort_values("support", ascending=False)
    export_itemsets = itemsets_table[["Products", "Items", "support"]].rename(columns={"support": "Support"}).copy()
    export_itemsets["Support"] = export_itemsets["Support"].map(lambda x: f"{x:.2%}")
    st.caption(f"Showing {len(export_itemsets):,} matching itemsets")
    st.dataframe(export_itemsets, width="stretch", hide_index=True, height=400)
    st.download_button(
        "Export frequent itemsets · CSV",
        data=export_itemsets.to_csv(index=False).encode("utf-8-sig"),
        file_name="frequent_itemsets.csv",
        mime="text/csv",
        key="export_itemsets_final",
    )

# ------------------------------ DATA LAB / METHOD CHECK ------------------------------

section_head("10 / Transparency", "See what went into the analysis", "Inspect the cleaned rows, basket encoding, and the consistency check for support, confidence, and lift.")
with st.expander("Source and cleaned data previews", expanded=False):
    st.markdown("**Uploaded source · first 20 rows**")
    # String conversion avoids mixed-object Arrow rendering problems with fields such as StockCode.
    st.dataframe(raw_data.head(20).astype(str), width="stretch", hide_index=True, height=280)
    st.markdown("**Cleaned purchase lines · first 20 rows**")
    st.dataframe(cleaned_data[["InvoiceNo", "Description", "Quantity"]].head(20).astype({"InvoiceNo": str, "Description": str}), width="stretch", hide_index=True, height=260)
    st.markdown("**Encoded basket matrix · first 10 invoices and 20 products**")
    st.caption("1 = purchased in this invoice · 0 = not present")
    st.dataframe(basket.iloc[:10, :20].astype(int).reset_index(), width="stretch", hide_index=True)

with st.expander("Algorithm and metric verification", expanded=False):
    if metrics_valid:
        st.success("PASS — confidence and lift agree with formula-based recomputations within floating-point tolerance.")
    else:
        st.error("FAIL — at least one confidence/lift value differs from its formula-based recomputation. Review the rule-generation step.")
    st.markdown(
        "- **Support:** fraction of analyzed invoices containing the itemset.\n"
        "- **Confidence A → B:** support(A ∪ B) divided by support(A).\n"
        "- **Lift A → B:** confidence(A → B) divided by support(B).\n"
        "- **Lift > 1:** positive association in this sample; it does not prove causation."
    )
    st.caption("The formula check verifies metric consistency; it is not a separate benchmark of Apriori against another library.")

footer()
