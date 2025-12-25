# app.py -- Fake News Detector (Aurora Neon: placeholders removed)
import streamlit as st
import joblib
import re
import io
import base64
import pandas as pd
import numpy as np
from time import sleep
import datetime
import os

# Optional article extraction libs (we'll try imports at runtime)
try:
    from newspaper import Article
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False

try:
    from readability import Document
    from bs4 import BeautifulSoup
    import requests
    _HAS_READABILITY = True
except Exception:
    _HAS_READABILITY = False

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Fake News Detector — Aurora Neon+", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------
# Theme configuration (CSS generator)
# ---------------------------
def make_css(theme_name: str):
    """Return CSS string adapted to theme_name."""
    # Minimal Light and Neo Glass unchanged from previous file (kept short)
    if theme_name == "Minimal Light":
        css = f"""
        <style>
        :root {{
          --accent:#0b5fff;
          --glass-bg: rgba(255,255,255,0.03);
          --card-bg: rgba(255,255,255,0.96);
          --text: rgba(8,8,8,0.95);
          --muted: rgba(40,40,40,0.55);
          --glass-border: rgba(0,0,0,0.06);
          --page-bg: linear-gradient(180deg,#f7f9fc 0%, #f1f5f9 100%);
        }}
        [data-testid="stAppViewContainer"] {{
           background: var(--page-bg);
           color: var(--text);
        }}
        .glass {{ background: var(--card-bg); color:var(--text); box-shadow: none; border:1px solid rgba(0,0,0,0.05); }}
        .logo {{ background: linear-gradient(135deg,#0b5fff,#00bcd4); color:white; }}
        .muted {{ color:var(--muted); }}
        .footer {{ color:var(--muted); }}
        </style>
        """
        return css

    if theme_name == "Neo Glass":
        css = f"""
        <style>
        :root {{
          --accent:#ff3d81;
          --glass-bg: rgba(18,7,30,0.65);
          --card-bg: rgba(20,12,30,0.7);
          --text: rgba(255,245,255,0.95);
          --muted: rgba(255,245,255,0.62);
          --glass-border: rgba(255,245,255,0.06);
        }}
        [data-testid="stAppViewContainer"] {{
           background: radial-gradient(circle at 10% 10%, rgba(255,61,129,0.06), transparent 10%),
                       linear-gradient(135deg,#05040a 0%, #071023 100%);
           color: var(--text);
        }}
        .glass {{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); color:var(--text); }}
        .logo {{ background: linear-gradient(135deg,#ff3d81,#7c4dff); color:white; }}
        .muted {{ color:var(--muted); }}
        </style>
        """
        return css

    # Aurora Neon (enhanced: colorful, animated background, multi-stop gradients)
    if theme_name == "Aurora Neon":
        css = """
        <style>
        :root{
          --accent-1: #7c00ff;
          --accent-2: #00ffc6;
          --accent-3: #ff3df7;
          --accent-4: #ffd500;
          --glass-bg: rgba(8,6,20,0.55);
          --card-bg: rgba(10,8,24,0.65);
          --text: #f8ffff;
          --muted: rgba(248,255,255,0.72);
        }

        /* animated multi-layer nebula background */
        @keyframes nebulaShift {
          0% { transform: translate3d(0,0,0) scale(1); opacity: 1; }
          50% { transform: translate3d(-6%,4%,0) scale(1.02); opacity: 0.98; }
          100% { transform: translate3d(0,0,0) scale(1); opacity: 1; }
        }

        [data-testid="stAppViewContainer"]{
          background-color: #04030b;
          background-image:
            radial-gradient(circle at 10% 10%, rgba(124,0,255,0.14), transparent 12%),
            radial-gradient(circle at 85% 80%, rgba(0,255,198,0.08), transparent 14%),
            linear-gradient(135deg, rgba(6,0,16,1), rgba(4,10,28,1));
          color: var(--text);
          overflow-x: hidden;
        }

        /* soft animated overlay layers for color movement */
        .__nebula_layer_1 {
          position: absolute;
          left: -10%;
          top: -8%;
          width: 60vw;
          height: 70vh;
          pointer-events: none;
          background: radial-gradient(circle, rgba(124,0,255,0.12) 0%, rgba(124,0,255,0.06) 20%, transparent 50%);
          filter: blur(72px);
          animation: nebulaShift 12s ease-in-out infinite;
          z-index: 0;
        }
        .__nebula_layer_2 {
          position: absolute;
          right: -8%;
          bottom: -6%;
          width: 55vw;
          height: 65vh;
          pointer-events: none;
          background: radial-gradient(circle, rgba(0,255,198,0.10) 0%, rgba(255,61,247,0.06) 25%, transparent 60%);
          filter: blur(86px);
          animation: nebulaShift 16s ease-in-out infinite reverse;
          z-index: 0;
        }

        main .block-container{
          position: relative;
          z-index: 10;
          padding-top: 1.5rem;
          padding-left: 2rem;
          padding-right: 2rem;
          padding-bottom: 2rem;
          max-width: 1100px;
          margin-left: auto;
          margin-right: auto;
        }

        .glass {
          background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
          border-radius: 14px;
          padding: 18px;
          border: 1px solid rgba(0,255,198,0.06);
          box-shadow: 0 14px 50px rgba(0,255,198,0.03);
          backdrop-filter: blur(8px) saturate(1.1);
          color: var(--text);
          margin-bottom: 14px;
        }

        .header-title{ display:flex; align-items:center; gap:14px; z-index:11; }
        .logo {
          width:56px;
          height:56px;
          border-radius: 12px;
          background: conic-gradient(from 180deg, #7c00ff, #ff3df7, #00ffc6, #ffd500, #7c00ff);
          display:flex;
          align-items:center;
          justify-content:center;
          font-weight:700;
          color:black;
          box-shadow: 0 16px 60px rgba(124,0,255,0.08), inset 0 -6px 18px rgba(0,0,0,0.12);
        }
        .stButton>button { border-radius: 12px; padding: 8px 14px; font-weight:700; border: none; background: linear-gradient(90deg,#ff3df7,#00ffc6); color: #041017; }
        .muted { color: var(--muted); font-size:0.92rem; }
        .footer { font-size:0.9rem; color:var(--muted); padding-top:8px; }

        .result h3, .result .muted { color: var(--text); }

        /* colorful badge shadow */
        .badge { box-shadow: 0 18px 60px rgba(124,0,255,0.06); border-radius: 16px; }

        /* small responsive tweaks */
        @media (max-width:800px){
          .logo { width:48px; height:48px; }
        }
        </style>
        """
        return css

    # Default Glass theme (unchanged)
    css = """
    <style>
    :root{
      --accent:#7c4dff;
      --glass-bg: rgba(255,255,255,0.04);
      --card-bg: rgba(255,255,255,0.03);
      --text: rgba(255,255,255,0.95);
      --muted: rgba(255,255,255,0.65);
      --glass-border: rgba(255,255,255,0.06);
    }

    [data-testid="stAppViewContainer"]{
      background: radial-gradient(circle at 10% 10%, rgba(124,77,255,0.12), transparent 10%),
                  radial-gradient(circle at 90% 90%, rgba(0,200,255,0.06), transparent 8%),
                  linear-gradient(135deg, #071021 0%, #061322 100%);
      color: var(--text);
    }

    main .block-container{
      padding-top: 1.5rem;
      padding-left: 2rem;
      padding-right: 2rem;
      padding-bottom: 2rem;
      max-width: 1100px;
      margin-left: auto;
      margin-right: auto;
    }

    .glass {
      background: var(--glass-bg);
      border-radius: 14px;
      padding: 18px;
      border: 1px solid var(--glass-border);
      box-shadow: 0 8px 30px rgba(2,6,23,0.6);
      backdrop-filter: blur(8px) saturate(1.1);
      color: var(--text);
      margin-bottom: 14px;
    }

    .header-title{
      display:flex;
      align-items:center;
      gap:14px;
    }
    .logo {
      width:56px;
      height:56px;
      border-radius: 12px;
      background: linear-gradient(135deg, rgba(124,77,255,0.95), rgba(0,200,255,0.85));
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      color:white;
      box-shadow: 0 6px 24px rgba(124,77,255,0.12), inset 0 -6px 18px rgba(0,0,0,0.12);
    }
    .stButton>button {
      border-radius: 10px;
      padding: 8px 14px;
      font-weight:600;
      border: none;
    }
    .muted { color: var(--muted); font-size:0.92rem; }
    .footer { font-size:0.9rem; color:var(--muted); padding-top:8px; }
    .result h3, .result .muted { color: var(--text); }
    </style>
    """
    return css

# ---------------------------
# Sidebar: theme + URL checker + quick settings
# ---------------------------
with st.sidebar:
    st.markdown('<div class="glass" style="padding:12px">', unsafe_allow_html=True)
    st.markdown('<div class="header-title"><div class="logo">FN</div><div>'
                '<h3 style="margin:0">FakeNews — Aurora+</h3>'
                '<div class="muted">AI Content Verification</div>'
                '</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Theme")
    theme_choice = st.selectbox("Pick theme", ["Glass (default)", "Minimal Light", "Neo Glass", "Aurora Neon"], index=3)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Real-time URL checker")
    url_input = st.text_input("Paste article URL here", placeholder="https://example.com/article...")
    fetch_btn = st.button("Fetch & Analyze", key="fetch_url")
    st.markdown("<small class='muted'>Uses newspaper3k or readability fallback.</small>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Info")
    st.markdown("This app uses Logistic Regression + TF-IDF.")
    st.markdown("Retrain using: `python f2.py`")
    st.markdown("</div>", unsafe_allow_html=True)

# Apply selected theme CSS
st.markdown(make_css(theme_choice), unsafe_allow_html=True)

# If Aurora Neon, inject nebula overlay divs so background animation shows
if theme_choice == "Aurora Neon":
    st.markdown("<div class='__nebula_layer_1'></div><div class='__nebula_layer_2'></div>", unsafe_allow_html=True)

# ---------------------------
# Load model + vectorizer
# ---------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        vect = joblib.load("vectorizer.pkl")
        return model, vect
    except Exception:
        st.error("Model files missing. Run f2.py first.")
        st.stop()

model, vectorizer = load_artifacts()

# ---------------------------
# Preprocessing setup
# ---------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------
# Layout (main)
# ---------------------------
left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("<h1 style='margin:0; font-size: 34px;'>🛡️ News Trust Validator</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted' style='font-size:15px;'>AI-Powered Authenticity Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Paste any article text below, or use the Real-time URL checker in the sidebar.</div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)

    user_text = st.text_area("Paste content", height=260, placeholder="Enter article text here...")
    col1, col2 = st.columns([1,1])
    with col1:
        predict_btn = st.button("Verify", key="single")
    with col2:
        clear_btn = st.button("Clear")

    if clear_btn:
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    # Batch upload area (unchanged)
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0'>Batch Analysis</h4>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Upload a CSV with a 'text' column.</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                st.info(f"Loaded {len(df)} rows — processing...")
                df["clean"] = df["text"].astype(str).apply(clean_text)
                X = vectorizer.transform(df["clean"])
                probs = model.predict_proba(X)
                preds = np.where(probs.argmax(axis=1)==1, "real", "fake")
                df["label"] = preds
                df["confidence"] = probs.max(axis=1)
                st.success("Completed.")
                st.dataframe(df.head(200))
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Error: " + str(e))
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0'>Live Analysis</h3>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Prediction & confidence</div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    result_holder = st.empty()
    gauge_holder  = st.empty()
    explain_holder = st.empty()
    source_holder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Helper: build interactive SVG gauge HTML (supports multi-stop gradients)
# ---------------------------
def build_svg_gauge(pct: float, size: int = 140, stroke_width: int = 14, label_text: str = "Score", gradient_stops=None):
    """
    gradient_stops: list of color hex strings e.g. ['#7c00ff','#00ffc6','#ff3df7']
    If None, uses default 2-stop gradients depending on variant set by caller.
    """
    pct = max(0.0, min(100.0, float(pct)))
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.141592653589793 * radius
    offset_target = circumference * (1 - pct / 100.0)

    # default gradient if not provided
    if not gradient_stops:
        gradient_stops = ["#34d399", "#10b981"]

    # create SVG linearGradient stops dynamically
    stops_svg = ""
    for i, color in enumerate(gradient_stops):
        offset = int((i / max(1, len(gradient_stops)-1)) * 100)
        stops_svg += f'<stop offset="{offset}%" stop-color="{color}" />'

    svg = f"""
    <div class="gauge-wrap" style="width:{size}px; height:{size}px;">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="gA" x1="0" x2="1">
            {stops_svg}
          </linearGradient>
        </defs>
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="rgba(255,255,255,0.06)" stroke-width="{stroke_width}" fill="none" />
        <circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="url(#gA)" stroke-width="{stroke_width}" fill="none"
                stroke-linecap="round"
                stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{circumference:.2f}">
          <animate attributeName="stroke-dashoffset" from="{circumference:.2f}" to="{offset_target:.2f}"
                   dur="700ms" fill="freeze" begin="0.08s" />
        </circle>
        <text x="50%" y="50%" dominant-baseline="central" text-anchor="middle"
              font-size="{size*0.18}" font-weight="700" fill="white">{pct:.1f}%</text>
        <text x="50%" y="{size*0.66}" dominant-baseline="central" text-anchor="middle"
              font-size="{size*0.095}" fill="rgba(255,255,255,0.75)">{label_text}</text>
      </svg>
    </div>
    """
    return svg

# ---------------------------
# Utility: fetch article text from URL
# ---------------------------
def fetch_article_text(url: str):
    """Try to extract article text and title from a URL.
       Returns (title, clean_text, source_domain) or raises Exception."""
    url = url.strip()
    if _HAS_NEWSPAPER:
        try:
            art = Article(url)
            art.download()
            art.parse()
            title = art.title or ""
            text = art.text or ""
            if text.strip():
                return title, text, art.source_url or url
        except Exception:
            # fall through to readability
            pass

    if _HAS_READABILITY:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/100 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        doc = Document(resp.text)
        title = doc.short_title() or ""
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        text = soup.get_text(separator="\n").strip()
        return title, text, resp.url

    # If neither lib present, try simple requests + BeautifulSoup (if bs4 available)
    try:
        import requests as _req
        from bs4 import BeautifulSoup as _BS
        headers = {"User-Agent": "Mozilla/5.0"}
        r = _req.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        s = _BS(r.text, "html.parser")
        # heuristic: join all <p> tags
        paragraphs = s.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        title = s.title.string if s.title else ""
        return title, text, r.url
    except Exception as e:
        raise Exception("Article extraction failed. Install newspaper3k or readability-lxml, or check URL.") from e

# ---------------------------
# Prediction function (core)
# ---------------------------
def analyze_text_and_render(article_text: str, source_title: str = None, source_url: str = None):
    """Runs preprocessing, prediction and renders staged UI + gauge."""
    cleaned = clean_text(article_text)
    X = vectorizer.transform([cleaned])
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    label = "real" if pred_idx == 1 else "fake"
    confidence = float(probs[pred_idx])  # 0..1

    # Show source info if available
    if source_title or source_url:
        info_html = "<div style='margin-bottom:8px'>"
        if source_title:
            info_html += f"<div style='font-weight:700'>{source_title}</div>"
        if source_url:
            info_html += f"<div class='muted' style='font-size:0.9rem'>{source_url}</div>"
        info_html += "</div>"
        source_holder.markdown(info_html, unsafe_allow_html=True)
    else:
        source_holder.empty()

    # Staged rendering (badge -> card -> svg gauge)
    if label == "real":
        title_html = "<h3 style='margin:0;color:#dfffe6'>Authentic Content</h3>"
        result_class = "result real"
    else:
        title_html = "<h3 style='margin:0;color:#ffe6e6'>Misleading / Fake</h3>"
        result_class = "result fake"

    sub_html = f"<div class='muted'>Credibility Score: {confidence*100:.2f}%</div>"

    # badge SVG (multicolor for Aurora Neon)
    if theme_choice == "Aurora Neon":
        if label == "real":
            svg_badge = """
            <svg width="78" height="78" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="multig1" x1="0" x2="1">
                  <stop offset="0%" stop-color="#7c00ff" />
                  <stop offset="40%" stop-color="#ff3df7" />
                  <stop offset="70%" stop-color="#00ffc6" />
                  <stop offset="100%" stop-color="#ffd500" />
                </linearGradient>
              </defs>
              <g>
                <circle cx="60" cy="60" r="54" fill="url(#multig1)" opacity="0.98" />
                <path d="M38 62 L54 78 L86 46" fill="none" stroke="white" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"
                      stroke-dasharray="120" stroke-dashoffset="120">
                  <animate attributeName="stroke-dashoffset" from="120" to="0" dur="420ms" fill="freeze" begin="0.05s" />
                </path>
              </g>
            </svg>
            """
        else:
            svg_badge = """
            <svg width="78" height="78" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="multir1" x1="0" x2="1">
                  <stop offset="0%" stop-color="#ffd500" />
                  <stop offset="33%" stop-color="#ff3df7" />
                  <stop offset="66%" stop-color="#7c00ff" />
                  <stop offset="100%" stop-color="#00ffc6" />
                </linearGradient>
              </defs>
              <g>
                <circle cx="60" cy="60" r="54" fill="url(#multir1)" opacity="0.98" />
                <g stroke="white" stroke-width="9" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="120" stroke-dashoffset="120">
                  <path d="M42 42 L78 78" fill="none">
                    <animate attributeName="stroke-dashoffset" from="120" to="0" dur="420ms" fill="freeze" begin="0.05s" />
                  </path>
                  <path d="M78 42 L42 78" fill="none">
                    <animate attributeName="stroke-dashoffset" from="120" to="0" dur="420ms" fill="freeze" begin="0.15s" />
                  </path>
                </g>
              </g>
            </svg>
            """
    else:
        # fallback badges (original)
        if label == "real":
            svg_badge = """
            <svg width="78" height="78" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="g1" x1="0" x2="1">
                  <stop offset="0" stop-color="#34d399"/>
                  <stop offset="1" stop-color="#10b981"/>
                </linearGradient>
              </defs>
              <g>
                <circle cx="60" cy="60" r="54" fill="url(#g1)" opacity="0.95" />
                <path d="M38 62 L54 78 L86 46" fill="none" stroke="white" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"
                      stroke-dasharray="120" stroke-dashoffset="120">
                  <animate attributeName="stroke-dashoffset" from="120" to="0" dur="380ms" fill="freeze" begin="0.06s" />
                </path>
              </g>
            </svg>
            """
        else:
            svg_badge = """
            <svg width="78" height="78" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="r1" x1="0" x2="1">
                  <stop offset="0" stop-color="#ff7a7a"/>
                  <stop offset="1" stop-color="#ff4d4d"/>
                </linearGradient>
              </defs>
              <g>
                <circle cx="60" cy="60" r="54" fill="url(#r1)" opacity="0.95" />
                <path d="M42 42 L78 78 M78 42 L42 78" fill="none" stroke="white" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"
                      stroke-dasharray="120" stroke-dashoffset="120">
                  <animate attributeName="stroke-dashoffset" from="120" to="0" dur="380ms" fill="freeze" begin="0.06s" />
                </path>
              </g>
            </svg>
            """

    # 1) badge pop (no decorative placeholder)
    badge_html = f"""
    <div class="glass" style="padding:12px; display:flex; align-items:center; gap:12px;">
      <div style="width:78px; height:78px;">{svg_badge}</div>
      <div style="display:flex;flex-direction:column;">
        <div style="font-weight:700">{'Authentic Content' if label=='real' else 'Misleading / Fake'}</div>
        <div class='muted' style='margin-top:6px;'>Scoring...</div>
      </div>
    </div>
    """
    result_holder.markdown(badge_html, unsafe_allow_html=True)
    sleep(0.14)

    # 2) full card
    card_html = f"""
    <div class="glass {'result real' if label=='real' else 'result fake'} animate" style="padding:14px; display:flex; gap:12px; align-items:center;">
        <div style="width:78px; height:78px;">{svg_badge}</div>
        <div style="display:flex;flex-direction:column; gap:6px;">
            {title_html}
            {sub_html}
        </div>
    </div>
    """
    result_holder.markdown(card_html, unsafe_allow_html=True)

    # 3) interactive gauge
    pct = confidence * 100.0
    # choose gradient stops:
    if theme_choice == "Aurora Neon":
        # vibrant multi-stop gradient
        grad_stops = ["#7c00ff", "#ff3df7", "#00ffc6", "#ffd500"]
    else:
        # default two-stop gradient (green/orange/red depends on pct)
        if pct < 40:
            grad_stops = ["#ff7a7a", "#ff4d4d"]
        elif pct < 70:
            grad_stops = ["#f59e0b", "#f97316"]
        else:
            grad_stops = ["#34d399", "#10b981"]

    gauge_html = build_svg_gauge(pct=pct, size=150, stroke_width=14, label_text="Credibility", gradient_stops=grad_stops)
    gauge_holder.markdown(gauge_html, unsafe_allow_html=True)

    # explain toggle
    explain_holder.empty()
    if st.button("Explain score", key=f"explain_{int(datetime.datetime.now().timestamp()*1000)}"):
        if pct >= 80:
            explain_text = ("The model is highly confident this content is authentic based on linguistic cues "
                            "and similarity to real-news examples in the training data.")
        elif pct >= 50:
            explain_text = ("The model finds some indicators of authenticity but also some patterns "
                            "that appear in both real and fake examples — treat result as uncertain.")
        else:
            explain_text = ("The model identifies several linguistic patterns typical of fabricated or "
                            "misleading content. Verify with trusted sources before sharing.")
        explain_holder.markdown(f"<div class='muted' style='margin-top:8px'>{explain_text}</div>", unsafe_allow_html=True)

    return {"label": label, "confidence": confidence, "pct": pct}

# ---------------------------
# Handle sidebar Fetch & Analyze
# ---------------------------
if fetch_btn and url_input.strip():
    # try fetching
    source_holder.empty()
    result_holder.empty()
    gauge_holder.empty()
    explain_holder.empty()
    try:
        with st.spinner("Fetching article..."):
            title, article_text, final_url = fetch_article_text(url_input)
            if not article_text or len(article_text.strip()) < 80:
                raise Exception("Extracted article is too short / empty.")
        # render analysis
        analyze_text_and_render(article_text=article_text, source_title=title, source_url=final_url)
    except Exception as e:
        st.error("Could not fetch/extract article: " + str(e))

# ---------------------------
# Handle manual pasted text analysis (main input)
# ---------------------------
if predict_btn:
    if not user_text.strip():
        st.warning("Please enter text.")
    else:
        analyze_text_and_render(article_text=user_text)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="footer muted">Powered by AI — Aurora Neon+ Edition</div>', unsafe_allow_html=True)
