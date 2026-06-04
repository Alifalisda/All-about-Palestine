import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="All-about-Palestine",
    layout="wide",
    initial_sidebar_state="auto"  # auto-collapse on mobile
)

# ─────────────────────────────────────────────
# GLOBAL RESPONSIVE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit footer & menu */
    #MainMenu, footer { visibility: hidden; }

    /* Responsive font scaling */
    html { font-size: clamp(12px, 2vw, 16px); }

    /* Responsive block container padding */
    .block-container {
        padding: clamp(0.5rem, 2vw, 2rem) !important;
        max-width: 100% !important;
    }

    /* Sidebar width responsive */
    [data-testid="stSidebar"] {
        min-width: 200px !important;
        max-width: 260px !important;
    }

    /* Responsive dataframe */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        overflow-x: auto !important;
    }

    /* Responsive charts */
    [data-testid="stpyplot"] {
        width: 100% !important;
    }

    /* Responsive iframe wrapper */
    .iframe-wrapper {
        position: relative;
        width: 100%;
        overflow: hidden;
    }

    /* Responsive title */
    h1 { font-size: clamp(1.2rem, 4vw, 2.5rem) !important; }
    h2 { font-size: clamp(1rem, 3vw, 1.8rem) !important; }
    h3 { font-size: clamp(0.9rem, 2.5vw, 1.4rem) !important; }

    /* Better button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-size: clamp(0.8rem, 1.5vw, 1rem);
    }

    /* Responsive text area */
    .stTextArea textarea {
        color: white;
        font-size: clamp(0.8rem, 1.5vw, 1rem);
    }

    /* Responsive slider */
    .stSlider { width: 100% !important; }

    /* Responsive multiselect */
    .stMultiSelect { width: 100% !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    try:
        df = pd.read_csv("reddit_opinion_PSE_ISR_1.csv")
        df.to_parquet("dataset.parquet")
        df = pd.read_parquet("dataset.parquet")
        for col in ['created_time', 'post_created_time', 'user_account_created_time']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
def show_home():
    st.components.v1.html(
        """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            .wrapper { width: 100%; }
            iframe { width: 100%; border: none; display: block; }
        </style>
        <div class="wrapper">
            <iframe
                id="home-frame"
                src="https://lookerstudio.google.com/embed/reporting/34102220-751f-4e6c-864f-f42ddd08ef39/page/JgD"
                allowfullscreen>
            </iframe>
        </div>
        <script>
            function resizeFrame() {
                const vh = window.innerHeight;
                const frame = document.getElementById('home-frame');
                frame.style.height = vh + 'px';
                if (window.frameElement) {
                    window.frameElement.style.height = vh + 'px';
                    window.frameElement.setAttribute('height', vh);
                }
            }
            resizeFrame();
            window.addEventListener('resize', resizeFrame);
        </script>
        """,
        height=700,
        scrolling=False
    )


# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
def show_history():
    st.title("The History")
    st.components.v1.html(
        """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            .wrapper { width: 100%; }
            iframe { width: 100%; border: none; display: block; }
        </style>
        <div class="wrapper">
            <iframe
                id="history-frame"
                src="https://datastudio.google.com/embed/reporting/34102220-751f-4e6c-864f-f42ddd08ef39/page/p_abraimownd"
                allowfullscreen>
            </iframe>
        </div>
        <script>
            function resizeFrame() {
                const vh = window.innerHeight;
                const titleOffset = 80;
                const frame = document.getElementById('history-frame');
                frame.style.height = (vh - titleOffset) + 'px';
                if (window.frameElement) {
                    window.frameElement.style.height = vh + 'px';
                    window.frameElement.setAttribute('height', vh);
                }
            }
            resizeFrame();
            window.addEventListener('resize', resizeFrame);
        </script>
        """,
        height=700,
        scrolling=False
    )


# ─────────────────────────────────────────────
# SENTIMENT ANALYSIS
# ─────────────────────────────────────────────
PRO_PALESTINE_KW = [
    "palestine", "gaza", "free palestine", "apartheid", "nakba",
    "zionist aggression", "ethnic cleansing", "end occupation",
    "save al-aqsa", "boycott israel", "humanitarian crisis in gaza",
    "illegal settlements", "palestinian solidarity", "massacres in palestine",
    "zionist crimes", "zionist"
]

PRO_ISRAEL_KW = [
    "hate israel", "idf", "hamas terrorism", "zionism", "jewish state",
    "defend israel", "iranian proxies", "rocket attacks", "security for israel",
    "stop hamas", "iran's threat to israel", "peace accords",
    "abraham accords", "holocaust remembrance", "justice for israel",
    "right to defend", "hamas aggression"
]


def assign_sentiment_category(row):
    title = row['post_title'].lower()
    score = row['post_sentiment']
    is_pal = any(kw in title for kw in PRO_PALESTINE_KW)
    is_isr = any(kw in title for kw in PRO_ISRAEL_KW)

    if is_pal:
        if score > 0.3:   return 'Positive (Pro-Palestine)'
        if score < -0.3:  return 'Negative (Pro-Palestine)'
        return 'Neutral (Pro-Palestine)'
    if is_isr:
        if score > 0.3:   return 'Positive (Pro-Israel)'
        if score < -0.3:  return 'Negative (Pro-Israel)'
        return 'Neutral (Pro-Israel)'
    if score > 0.3:  return 'Positive'
    if score < -0.3: return 'Negative'
    return 'Neutral'


def show_sentiment_analysis(df):
    st.title("Sentiment Analysis")
    st.caption("The world needs more justice and less war. "
               "Let's stand together for human rights.")

    # ── Preprocessing ──────────────────────────────
    df['post_title'] = df['post_title'].fillna('')
    df['post_sentiment'] = df['post_title'].apply(
        lambda t: TextBlob(t).sentiment.polarity if t else 0
    )
    df['post_sentiment_category'] = df.apply(assign_sentiment_category, axis=1)

    # ── Sidebar filters ────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        sentiment_filter = st.multiselect(
            "Sentiment Category",
            options=sorted(df['post_sentiment_category'].unique()),
            default=list(df['post_sentiment_category'].unique())
        )

    filtered = df[df['post_sentiment_category'].isin(sentiment_filter)]

    # ── KPI cards ──────────────────────────────────
    total    = len(filtered)
    positive = len(filtered[filtered['post_sentiment_category'].str.startswith('Positive')])
    negative = len(filtered[filtered['post_sentiment_category'].str.startswith('Negative')])
    neutral  = len(filtered[filtered['post_sentiment_category'].str.startswith('Neutral')])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Posts", f"{total:,}")
    c2.metric("Positive", f"{positive:,}")
    c3.metric("Negative", f"{negative:,}")
    c4.metric("Neutral",  f"{neutral:,}")

    st.divider()

    # ── Dataframe ──────────────────────────────────
    with st.expander("📄 View Data", expanded=False):
        drop_cols = [c for c in ['comment_id','score','post_id','controversiality',
                                  'user_account_created_time','post_upvote_ratio',
                                  'post_thumbs_ups','post_created_time'] if c in filtered.columns]
        st.dataframe(filtered.drop(columns=drop_cols), use_container_width=True)

    # ── Charts: 2-column on desktop, 1-column on mobile ──
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("Top Subreddits")
        top_sub = df['subreddit'].value_counts().head(15).reset_index()
        top_sub.columns = ['subreddit', 'count']
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.barplot(x='count', y='subreddit', data=top_sub, palette='magma', ax=ax1)
        ax1.set_title('Top 15 Subreddits')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('')
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col_right:
        st.subheader("Sentiment Distribution")
        sent_counts = filtered['post_sentiment_category'].value_counts()
        total_s     = sent_counts.sum()
        pct         = (sent_counts / total_s) * 100
        combined    = sent_counts[pct >= 2].copy()
        others      = sent_counts[pct < 2].sum()
        if others > 0:
            combined["Others (<2%)"] = others

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.pie(
            combined,
            labels=combined.index,
            autopct='%1.0f%%',
            explode=[0.03] * len(combined),
            startangle=90,
            textprops={'fontsize': 7},
            labeldistance=1.08
        )
        ax2.set_title('Sentiment Distribution | Posts')
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.divider()

    # ── WordCloud ──────────────────────────────────
    st.subheader("WordCloud by Date")
    df['created_time'] = pd.to_datetime(df['created_time'])
    min_d = df['created_time'].min().date()
    max_d = df['created_time'].max().date()

    selected_date = st.slider(
        "Select date",
        min_value=min_d, max_value=max_d, value=min_d,
        format="YYYY-MM-DD"
    )

    day_df  = df[df['created_time'].dt.date == selected_date]
    text    = " ".join(day_df['self_text'].dropna())

    if text.strip():
        wc  = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis("off")
        ax3.set_title(f"WordCloud — {selected_date}", fontsize=12)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    else:
        st.info(f"No text data available for {selected_date}.")


# ─────────────────────────────────────────────
# CHECK REDDIT SENTIMENT
# ─────────────────────────────────────────────
def check_reddit_sentiment():
    st.title("Check Your Reddit Sentiment")
    st.caption("Input your Reddit post or comment to analyze its sentiment.")

    user_input = st.text_area("Enter your Reddit text here:", height=150)

    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            score = TextBlob(user_input).sentiment.polarity
            subj  = TextBlob(user_input).sentiment.subjectivity

            if score > 0.3:
                label, color, icon = "Positive 😊", "green", "✅"
            elif score < -0.3:
                label, color, icon = "Negative 😔", "red", "⚠️"
            else:
                label, color, icon = "Neutral 😐", "gray", "ℹ️"

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{icon} Sentiment", label)
            c2.metric("Polarity Score", f"{score:.2f}", help="-1 (negative) to +1 (positive)")
            c3.metric("Subjectivity", f"{subj:.2f}", help="0 (objective) to 1 (subjective)")

            # Visual bar
            st.progress(
                int((score + 1) / 2 * 100),
                text=f"Sentiment polarity: {score:.2f}"
            )
        else:
            st.warning("Please enter some text before analyzing.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_data()

    # Preprocessing
    if not df.empty:
        start_date = pd.to_datetime('2023-10-07')
        df = df.dropna(subset=['created_time', 'post_created_time'])
        df = df[(df['post_created_time'] >= start_date) & (df['created_time'] >= start_date)]

        if 'post_title' in df.columns:
            df = df[df['post_title'].notna()]
            df = df.drop_duplicates(subset=['post_title'])
            df['post_sentiment'] = df['post_title'].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity
            )
            df['post_sentiment_category'] = df['post_sentiment'].apply(
                lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral')
            )

    # Sidebar navigation
    with st.sidebar:
        app = option_menu(
            menu_title="All-about-Palestine",
            options=["Home", "History", "Sentiment Analysis"],
            icons=["house", "clock-history", "graph-up-arrow"],
            styles={
                "container": {"padding": "5!important"},
                "icon": {"color": "orange"},
                "nav-link": {"font-size": "14px"},
            }
        )

    # Routing
    if app == "Home":
        show_home()

    elif app == "History":
        show_history()

    elif app == "Sentiment Analysis":
        with st.sidebar:
            sentiment_menu = option_menu(
                menu_title="Choose an option:",
                options=["Analysis Data", "Check Your Reddit"],
                icons=["bar-chart", "chat-text"],
                styles={
                    "container": {"padding": "5!important"},
                    "icon": {"color": "orange"},
                    "nav-link": {"font-size": "14px"}
                }
            )
        if sentiment_menu == "Analysis Data":
            show_sentiment_analysis(df)
        elif sentiment_menu == "Check Your Reddit":
            check_reddit_sentiment()


if __name__ == "__main__":
    main()
