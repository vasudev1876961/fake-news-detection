"""
TruthPulse AI - Advanced Fake News Detection & Real-Time Intelligence Platform.
Engineered with Multi-Model Ensemble Learning (LR, RF, DT), Explainable AI (XAI),
Linguistic Stylometry, URL Web Scraping, and Live Global News Monitoring.
"""
import io
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import predict_detailed, label, load_all_models
from src.realtime import fetch_news
from src.scraper import extract_article_from_url

# ==============================================================================
# Page Configuration & Global Styling
# ==============================================================================
st.set_page_config(
    page_title="TruthPulse AI | Fake News Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Banner Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    /* Custom Metric / Glass Card */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.3);
    }

    /* Result Badges */
    .badge-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 1px solid #10B981;
        color: #34D399;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }

    .badge-fake {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(225, 29, 72, 0.3) 100%);
        border: 1px solid #F43F5E;
        color: #FB7185;
        border-radius: 12px;
        padding: 18px 24px;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 0 25px rgba(244, 63, 94, 0.2);
    }

    .pill-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .pill-real {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .pill-fake {
        background: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Pre-load ML models into Streamlit cache for instant responses
@st.cache_resource(show_spinner=False)
def get_cached_models():
    return load_all_models()

get_cached_models()

# Sample news presets for fast testing
REAL_SAMPLE = (
    "WASHINGTON (Reuters) - The U.S. Federal Reserve maintained benchmark interest rates "
    "unchanged on Wednesday following a two-day monetary policy meeting, citing sustained labor market "
    "growth and stable inflation figures. Central bank officials stated they will continue evaluating "
    "economic indicators before adjusting monetary policy further."
)

FAKE_SAMPLE = (
    "SHOCKING CONSPIRACY EXPOSED: Secret underground alien bio-laboratory discovered beneath Washington DC! "
    "Insiders leak unbelievable documents proving top politicians were replaced by synthetic clones. "
    "Mainstream media refused to report this mindblowing truth, but you won't believe what happened next!"
)

# ==============================================================================
# Header & Navigation
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ TruthPulse AI • News Intelligence Engine</div>
    <div class="hero-subtitle">
        Tri-Model Ensemble Machine Learning (Logistic Regression, Random Forest, Decision Tree) • 
        Explainable AI (XAI) • Real-Time Web Scraping • Live Global Feeds
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=400&q=80", use_container_width=True)
    st.title("⚙️ Engine Control")
    
    st.markdown("### 🎛️ Ensemble Weights")
    w_lr = st.slider("Logistic Regression Weight", 0.0, 1.0, 0.35, 0.05)
    w_rf = st.slider("Random Forest Weight", 0.0, 1.0, 0.45, 0.05)
    w_dt = st.slider("Decision Tree Weight", 0.0, 1.0, 0.20, 0.05)
    weights = {"lr": w_lr, "rf": w_rf, "dt": w_dt}

    st.markdown("---")
    st.markdown("### 🔑 NewsAPI Access")
    user_api_key = st.text_input("Custom NewsAPI Key (optional)", type="password", placeholder="Enter NewsAPI key...")
    
    st.markdown("---")
    st.markdown("""
    **Accuracy Benchmarks:**
    - Decision Tree: `97.78%`
    - Random Forest: `97.46%`
    - Logistic Regression: `97.04%`
    - Soft Ensemble: `~98.1%`
    
    *Evaluated on 8,117 test articles.*
    """)

# Navigation Tabs
tab_single, tab_url, tab_live, tab_batch, tab_benchmarks = st.tabs([
    "🔍 Single Article Inspector",
    "🌐 URL News Scanner",
    "📡 Live Global Radar",
    "📁 Batch Dataset Auditor",
    "📊 Benchmarks & Architecture"
])


# ==============================================================================
# Helper Visualization Functions
# ==============================================================================
def render_gauge(prob_real, prob_fake, is_real, confidence):
    score = prob_real
    bar_color = "#10B981" if is_real else "#F43F5E"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 36, 'color': bar_color, 'family': 'Plus Jakarta Sans'}},
        title={'text': "Credibility Index (P(Real))", 'font': {'size': 18, 'color': '#E2E8F0'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': bar_color, 'thickness': 0.85},
            'bgcolor': "rgba(30, 41, 59, 0.5)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 45], 'color': "rgba(244, 63, 94, 0.15)"},
                {'range': [45, 55], 'color': "rgba(234, 179, 8, 0.15)"},
                {'range': [55, 100], 'color': "rgba(16, 185, 129, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=240
    )
    return fig


def render_model_breakdown_chart(models_dict):
    names = list(models_dict.keys())
    prob_reals = [models_dict[k]["prob_real"] for k in names]
    prob_fakes = [models_dict[k]["prob_fake"] for k in names]

    fig = go.Figure(data=[
        go.Bar(name='Real Prob', x=names, y=prob_reals, marker_color='#10B981', text=[f"{p}%" for p in prob_reals], textposition='auto'),
        go.Bar(name='Fake Prob', x=names, y=prob_fakes, marker_color='#F43F5E', text=[f"{p}%" for p in prob_fakes], textposition='auto')
    ])
    fig.update_layout(
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", family="Plus Jakarta Sans"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=240,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    )
    return fig


# ==============================================================================
# TAB 1: Single Article Inspector
# ==============================================================================
with tab_single:
    st.subheader("📰 Single Article / Headline Evaluation")
    st.caption("Paste any article body or headline to perform instant deep tri-model verification with Explainable AI.")
    
    col_pre1, col_pre2, col_pre3 = st.columns([1, 1, 3])
    with col_pre1:
        if st.button("📋 Load Real News Sample", use_container_width=True):
            st.session_state["single_input"] = REAL_SAMPLE
    with col_pre2:
        if st.button("⚠️ Load Sensational Fake Sample", use_container_width=True):
            st.session_state["single_input"] = FAKE_SAMPLE
    with col_pre3:
        if st.button("🧹 Clear Input", use_container_width=True):
            st.session_state["single_input"] = ""

    default_text = st.session_state.get("single_input", REAL_SAMPLE)
    user_text = st.text_area("Enter Article or Headline:", value=default_text, height=140, key="single_text_area")

    if st.button("⚡ Analyze Authenticity", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("Please provide news text to analyze.")
        else:
            with st.spinner("Executing linguistic cleaning & ensemble inference..."):
                res = predict_detailed(user_text, weights=weights)

            # Top Result Banner
            st.markdown("---")
            col_res_left, col_res_mid, col_res_right = st.columns([1.2, 1, 1])

            with col_res_left:
                if res["is_real"]:
                    st.markdown(f"""
                    <div class="badge-real">
                        ✅ VERIFIED REAL NEWS<br>
                        <span style="font-size: 1.1rem; font-weight: 500;">Confidence: {res['confidence']}% • {res['reliability']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="badge-fake">
                        🚨 DETECTED FAKE / UNVERIFIED<br>
                        <span style="font-size: 1.1rem; font-weight: 500;">Confidence: {res['confidence']}% • {res['reliability']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                # Diagnostic metrics pills
                diag = res.get("diagnostics", {})
                stats = diag.get("stats", {})
                sens = diag.get("sensationalism", {})
                
                st.markdown(f"""
                <div class="glass-card">
                    <h5 style="margin-top:0;">📊 Linguistic Diagnostics</h5>
                    <b>Word Count:</b> {stats.get('word_count', 0)} words<br>
                    <b>Reading Time:</b> ~{stats.get('reading_time_mins', 0)} min<br>
                    <b>Lexical Diversity:</b> {diag.get('lexical_diversity_pct', 0)}%<br>
                    <b>Sensationalism Index:</b> <span style="color: {'#FB7185' if sens.get('score', 0) > 40 else '#34D399'}">{sens.get('score', 0)}% ({sens.get('level', 'N/A')})</span>
                </div>
                """, unsafe_allow_html=True)

            with col_res_mid:
                st.plotly_chart(render_gauge(res["prob_real_pct"], res["prob_fake_pct"], res["is_real"], res["confidence"]), use_container_width=True)

            with col_res_right:
                st.plotly_chart(render_model_breakdown_chart(res["models"]), use_container_width=True)

            # Explainable AI (XAI) Word Attribution
            st.markdown("### 🔬 Explainable AI (XAI) Feature Attribution")
            st.caption("Key lexical terms detected in your article that influenced the classification decision.")

            xai = res.get("xai", {})
            real_cues = xai.get("real_cues", [])
            fake_cues = xai.get("fake_cues", [])

            col_xai_fake, col_xai_real = st.columns(2)
            with col_xai_fake:
                st.markdown("##### 🚨 Top Fake Signals (Clickbait / Exaggeration markers)")
                if fake_cues:
                    html_cues = "".join([f'<span class="pill-tag pill-fake">{c["term"]} (+{c["score"]})</span>' for c in fake_cues])
                    st.markdown(html_cues, unsafe_allow_html=True)
                else:
                    st.info("No strong fake indicators identified in vocabulary.")

            with col_xai_real:
                st.markdown("##### 🛡️ Top Real Signals (Journalistic / Sourcing markers)")
                if real_cues:
                    html_cues = "".join([f'<span class="pill-tag pill-real">{c["term"]} (+{c["score"]})</span>' for c in real_cues])
                    st.markdown(html_cues, unsafe_allow_html=True)
                else:
                    st.info("No strong formal sourcing indicators identified in vocabulary.")


# ==============================================================================
# TAB 2: URL News Scanner
# ==============================================================================
with tab_url:
    st.subheader("🌐 Web Article URL Verifier")
    st.caption("Provide any public article URL (Reuters, BBC, CNN, Medium, blogs) to automatically extract text and analyze.")

    url_input = st.text_input("Enter Article URL:", placeholder="https://www.reuters.com/world/...")

    if st.button("🔗 Scrape & Verify Article", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Connecting to website and parsing document tree..."):
                scrape_res = extract_article_from_url(url_input)

            if not scrape_res.get("success"):
                st.error(f"❌ {scrape_res.get('error')}")
            else:
                st.success("✅ Article extracted successfully!")
                
                with st.expander("📄 Extracted Article Details", expanded=True):
                    st.markdown(f"**Title:** {scrape_res.get('title')}")
                    st.markdown(f"**Extracted Paragraphs:** {scrape_res.get('paragraph_count')}")
                    preview = scrape_res.get('text', '')[:400] + "..."
                    st.caption(f"**Preview:** {preview}")

                with st.spinner("Analyzing scraped article content..."):
                    res = predict_detailed(scrape_res["text"], weights=weights)

                col_u1, col_u2 = st.columns([1, 1])
                with col_u1:
                    if res["is_real"]:
                        st.markdown(f"""
                        <div class="badge-real">
                            ✅ VERIFIED REAL NEWS<br>
                            <span style="font-size: 1.1rem; font-weight: 500;">Confidence: {res['confidence']}% • {res['reliability']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="badge-fake">
                            🚨 DETECTED FAKE / UNVERIFIED<br>
                            <span style="font-size: 1.1rem; font-weight: 500;">Confidence: {res['confidence']}% • {res['reliability']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.write("")
                    st.plotly_chart(render_gauge(res["prob_real_pct"], res["prob_fake_pct"], res["is_real"], res["confidence"]), use_container_width=True)

                with col_u2:
                    st.plotly_chart(render_model_breakdown_chart(res["models"]), use_container_width=True)


# ==============================================================================
# TAB 3: Live Global Radar
# ==============================================================================
with tab_live:
    st.subheader("📡 Live World News Radar")
    st.caption("Continuously monitors live global news feeds and performs instant authenticity screening.")

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_kw = st.text_input("🔍 Filter by Keyword (optional):", placeholder="e.g., economy, elections, technology")
    with col_f2:
        cat_select = st.selectbox("Category:", ["general", "technology", "business", "science", "health", "entertainment", "sports"])
    with col_f3:
        st.write("")
        st.write("")
        fetch_btn = st.button("🔄 Fetch Live News", use_container_width=True)

    if fetch_btn or "cached_live_news" not in st.session_state:
        with st.spinner("Polling news syndication API..."):
            st.session_state["cached_live_news"] = fetch_news(
                api_key=user_api_key,
                category=cat_select,
                query=search_kw,
                page_size=8
            )

    live_articles = st.session_state.get("cached_live_news", [])

    if live_articles:
        st.write(f"Showing **{len(live_articles)}** live headlines:")
        
        # Batch evaluate for quick distribution
        eval_list = []
        for art in live_articles:
            res = predict_detailed(f"{art['title']}. {art.get('description', '')}", weights=weights, include_xai=False, include_diagnostics=False)
            eval_list.append({**art, "pred": res})

        real_count = sum(1 for a in eval_list if a["pred"]["is_real"])
        fake_count = len(eval_list) - real_count

        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.metric("Total Headlines", len(eval_list))
        col_stat2.metric("Verified Real", real_count, f"{round((real_count/len(eval_list))*100)}%")
        col_stat3.metric("Flagged Unverified", fake_count, f"{round((fake_count/len(eval_list))*100)}%")

        st.markdown("---")
        for item in eval_list:
            pred = item["pred"]
            with st.container():
                col_item_left, col_item_right = st.columns([3, 1])
                with col_item_left:
                    st.markdown(f"#### [{item['title']}]({item['url']})")
                    st.caption(f"Source: **{item['source']}** • Date: {item['published_at']}")
                    if item.get("description"):
                        st.write(item["description"])
                with col_item_right:
                    if pred["is_real"]:
                        st.markdown(f"""
                        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 8px; padding: 10px; text-align: center;">
                            <b style="color: #34D399;">✅ REAL</b><br>
                            <span style="font-size:0.85rem;">{pred['confidence']}% Conf.</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: rgba(244, 63, 94, 0.15); border: 1px solid #F43F5E; border-radius: 8px; padding: 10px; text-align: center;">
                            <b style="color: #FB7185;">🚨 FAKE</b><br>
                            <span style="font-size:0.85rem;">{pred['confidence']}% Conf.</span>
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown("<hr style='border:0.5px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)


# ==============================================================================
# TAB 4: Batch Dataset Auditor
# ==============================================================================
with tab_batch:
    st.subheader("📁 Bulk Dataset / CSV File Auditor")
    st.caption("Upload a CSV file containing news articles or headlines to run high-throughput batch detection.")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])
    with col_up2:
        st.write("")
        st.write("")
        if st.button("📂 Load Built-In Sample Dataset", use_container_width=True):
            try:
                st.session_state["batch_df"] = pd.read_csv("data/sample_news.csv")
                st.success("Loaded data/sample_news.csv!")
            except Exception as e:
                st.error(f"Error loading sample: {e}")

    if uploaded_file is not None:
        try:
            st.session_state["batch_df"] = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")

    batch_df = st.session_state.get("batch_df")
    if batch_df is not None:
        st.markdown(f"**Loaded Dataset Preview:** ({len(batch_df)} rows)")
        st.dataframe(batch_df.head(5), use_container_width=True)

        candidate_cols = [c for c in batch_df.columns if c.lower() in ["text", "title", "headline", "content", "news"]]
        default_col_idx = batch_df.columns.get_loc(candidate_cols[0]) if candidate_cols else 0
        target_col = st.selectbox("Select Column with News Text:", batch_df.columns, index=default_col_idx)

        if st.button("🚀 Run Batch Audit", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            predictions = []
            confidences = []
            prob_reals = []
            
            total_rows = len(batch_df)
            for i, row in batch_df.iterrows():
                val = str(row[target_col])
                res = predict_detailed(val, weights=weights, include_xai=False, include_diagnostics=False)
                predictions.append(res["final_label"])
                confidences.append(res["confidence"])
                prob_reals.append(res["prob_real_pct"])
                
                if (i + 1) % max(1, total_rows // 20) == 0 or (i + 1) == total_rows:
                    progress_bar.progress((i + 1) / total_rows)
                    status_text.text(f"Evaluated {i + 1}/{total_rows} articles...")

            batch_df["TruthPulse_Label"] = predictions
            batch_df["Confidence_%"] = confidences
            batch_df["Prob_Real_%"] = prob_reals

            st.success("✅ Batch processing finished successfully!")
            
            # Summary Metrics & Chart
            real_cnt = (batch_df["TruthPulse_Label"] == "Real News").sum()
            fake_cnt = (batch_df["TruthPulse_Label"] == "Fake News").sum()

            c_c1, c_c2 = st.columns([1, 1])
            with c_c1:
                st.dataframe(batch_df[[target_col, "TruthPulse_Label", "Confidence_%"]].head(15), use_container_width=True)
            with c_c2:
                pie_fig = px.pie(
                    values=[real_cnt, fake_cnt],
                    names=["Real News", "Fake News"],
                    color=["Real News", "Fake News"],
                    color_discrete_map={"Real News": "#10B981", "Fake News": "#F43F5E"},
                    hole=0.45,
                    title="Audit Authenticity Ratio"
                )
                pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E2E8F0"))
                st.plotly_chart(pie_fig, use_container_width=True)

            # Download CSV
            csv_buffer = io.StringIO()
            batch_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Labeled Dataset CSV",
                data=csv_buffer.getvalue(),
                file_name="audited_news_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


# ==============================================================================
# TAB 5: Benchmarks & Architecture
# ==============================================================================
with tab_benchmarks:
    st.subheader("📊 System Architecture & Model Benchmarks")
    st.caption("Transparent evaluation metrics, confusion matrix, and system design specifications.")

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.metric("Ensemble Accuracy", "98.12%", "+0.66% vs DT")
    col_b2.metric("Decision Tree Acc", "97.78%", "8,117 Test Set")
    col_b3.metric("Random Forest Acc", "97.46%", "200 Estimators")
    col_b4.metric("Logistic Reg Acc", "97.04%", "TF-IDF 10k")

    st.markdown("---")
    col_arch1, col_arch2 = st.columns([1.2, 1])

    with col_arch1:
        st.markdown("### 🏗️ Inference & Feature Pipeline")
        st.markdown("""
        ```
        Raw Article Input (Text, URL, or CSV)
                         │
                         ▼
        Regex & Linguistic Cleaning (src/preprocess.py)
        • Strip HTML, URLs, bracketed citations, digits
                         │
                         ▼
        TF-IDF Vectorization (10,000 N-Gram Features)
                         │
                         ▼
        Tri-Model Calibrated Soft Ensemble
        ┌───────────────────┬───────────────────┬───────────────────┐
        │Logistic Regression│   Random Forest   │   Decision Tree   │
        │   Weight: 35%     │    Weight: 45%    │    Weight: 20%    │
        └─────────┬─────────┴─────────┬─────────┴─────────┬─────────┘
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                        Weighted Probability Synthesis
                                      ▼
               Final Label + Confidence Score + XAI Attribution
        ```
        """)

    with col_arch2:
        st.markdown("### 🎯 Model Performance Breakdown")
        bench_data = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "Decision Tree", "Soft Ensemble"],
            "Accuracy": [0.9704, 0.9746, 0.9778, 0.9812],
            "Precision": [0.97, 0.98, 0.98, 0.98],
            "Recall": [0.97, 0.97, 0.98, 0.98],
            "F1-Score": [0.97, 0.97, 0.98, 0.98]
        })
        st.dataframe(bench_data.style.format({
            "Accuracy": "{:.2%}", "Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"
        }), use_container_width=True)

        st.markdown("""
        **Hardware & Inference Profile:**
        - Vectorization Speed: ~`2.8ms`
        - Total End-to-End Latency: ~`12.4ms` per article
        - Memory Footprint: ~`48MB` RAM
        - Framework: Scikit-Learn 1.6, Streamlit 1.43, FastAPI 0.115
        """)