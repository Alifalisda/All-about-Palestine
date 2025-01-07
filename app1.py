import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud



# Konfigurasi halaman Streamlit
st.set_page_config(page_title="All-about-Palestine", layout="wide")

# Fungsi untuk memuat data
def load_data():
    try:
        df = pd.read_csv("reddit_opinion_PSE_ISR_1.csv")
        df.to_parquet("dataset.parquet")
        df = pd.read_parquet("dataset.parquet")
        df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')
        df['post_created_time'] = pd.to_datetime(df['post_created_time'], errors='coerce')
        df['user_account_created_time'] = pd.to_datetime(df['user_account_created_time'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Fungsi untuk menampilkan beranda
def show_home():
    st.markdown("""
        <iframe 
            src="https://lookerstudio.google.com/embed/reporting/34102220-751f-4e6c-864f-f42ddd08ef39/page/JgD" 
            width="100%" height="800px" style="border:none;" allowfullscreen></iframe>
    """, unsafe_allow_html=True)

# Fungsi untuk menampilkan sejarah
def show_history():
    st.title("The History")
    st.markdown("""
        <iframe 
            src="https://lookerstudio.google.com/embed/reporting/b2543f86-9716-44e3-93a1-a792573a9c87/page/p_abraimownd" 
            width="100%" height="800px" style="border:none;" allowfullscreen></iframe>
    """, unsafe_allow_html=True)

# Fungsi untuk menampilkan analisis sentimen
def show_sentiment_analysis(df):
    st.title("Sentiment Analysis")
    st.text("The world needs more justice and less war. Let's stand together for human rights")
    
    # Kata kunci pro-Palestina dan pro-Israel
    pro_palestine_keywords = [
    "palestine", "gaza", "free palestine", "apartheid", "nakba", "zionist aggression", 
    "ethnic cleansing", "end occupation", "save al-aqsa", "boycott israel", 
    "humanitarian crisis in gaza", "illegal settlements", "palestinian solidarity", 
    "massacres in palestine", "zionist crimes", "zionist"
    ]

    pro_israel_keywords = [
        "hate israel", "idf", "hamas terrorism", "zionism", "jewish state", "defend israel", 
        "iranian proxies", "rocket attacks", "security for israel", "stop hamas", 
        "iran's threat to israel", "peace accords", "abraham accords", "holocaust remembrance",
        "justice for israel", "right to defend", "hamas aggression"
    ]

    # Pastikan kolom post_title tidak memiliki nilai None/NaN
    df['post_title'] = df['post_title'].fillna('')  # Mengganti NaN dengan string kosong

    # Menambahkan kolom post_sentiment (skor sentimen) menggunakan TextBlob
    df['post_sentiment'] = df['post_title'].apply(
        lambda text: TextBlob(text).sentiment.polarity if text else 0  # Jika teks kosong, skor sentimen 0
    )

    # Menambahkan kategori berdasarkan kata kunci dan nilai sentimen
    df['post_sentiment_category'] = df.apply(
    lambda row: (
        # Jika ada kata kunci pro-Palestine
        'Positive (Pro-Palestine)' if any(keyword in row['post_title'].lower() for keyword in pro_palestine_keywords) and row['post_sentiment'] > 0.3 else
        # Jika ada kata kunci pro-Israel
        'Positive (Pro-Israel)' if any(keyword in row['post_title'].lower() for keyword in pro_israel_keywords) and row['post_sentiment'] > 0.3 else
        # Jika ada kata kunci pro-Palestine
        'Negative (Pro-Palestine)' if any(keyword in row['post_title'].lower() for keyword in pro_palestine_keywords) and row['post_sentiment'] < -0.3 else
        # Jika ada kata kunci pro-Israel
        'Negative (Pro-Israel)' if any(keyword in row['post_title'].lower() for keyword in pro_israel_keywords) and row['post_sentiment'] < -0.3 else
        # Jika ada kata kunci pro-Palestine
        'Neutral (Pro-Palestine)' if any(keyword in row['post_title'].lower() for keyword in pro_palestine_keywords) and -0.3 <= row['post_sentiment'] <= 0.3 else
        # Jika ada kata kunci pro-Israel
        'Neutral (Pro-Israel)' if any(keyword in row['post_title'].lower() for keyword in pro_israel_keywords) and -0.3 <= row['post_sentiment'] <= 0.3 else
        # Fallback ke nilai sentimen jika tidak ada kata kunci
        'Positive' if row['post_sentiment'] > 0.3 else
        'Negative' if row['post_sentiment'] < -0.3 else
        'Neutral'
    ),
    axis=1
)
    # Filter sentimen
    sentiment_filter = st.sidebar.multiselect(
        "Select Sentiment to Display:",
        options=df['post_sentiment_category'].unique(),
        default=df['post_sentiment_category'].unique()
    )
    filtered_data = df[df['post_sentiment_category'].isin(sentiment_filter)]
    dataframe = filtered_data.drop(columns=['comment_id', 'score', 'post_id','controversiality',
                                        'user_account_created_time', 'post_upvote_ratio','post_thumbs_ups','post_created_time'])
    st.dataframe(dataframe)

    # Bar chart: Popular subreddits
    st.subheader("Top Popular Subreddits")
    fig, ax = plt.subplots(figsize=(10, 6))
    popular_subreddits = df['subreddit'].value_counts().reset_index()
    popular_subreddits.columns = ['subreddit', 'count']
    sns.barplot(x='count', y='subreddit', data=popular_subreddits, palette='magma', ax=ax)
    ax.set_title('Top Popular Subreddits')
    ax.set_xlabel('Count')
    ax.set_ylabel('Subreddit')
    st.pyplot(fig)

    # Pie chart: Sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(7,7),dpi=200)
    sentiment_counts = filtered_data['post_sentiment_category'].value_counts()
    # Menghitung distribusi sentimen
    total = sentiment_counts.sum()

    # Menghitung persentase
    sentiment_percentages = (sentiment_counts / total) * 100

    # Gabungkan kategori dengan persentase < 1% menjadi "Others"
    sentiment_counts_combined = sentiment_counts[sentiment_percentages >= 2]
    others_count = sentiment_counts[sentiment_percentages < 2].sum()

    # Tambahkan kategori "Others" jika ada
    if others_count > 0:
        sentiment_counts_combined["Others(percentages < 2%)"] = others_count
    
    # Membuat pie chart
    ax.pie(
        sentiment_counts_combined,
        labels=sentiment_counts_combined.index,
        autopct='%1.0f%%',
        explode=[0.03] * len(sentiment_counts_combined),
        startangle=90,
        textprops={'fontsize': 5},  # Ukuran font label lebih kecil
        labeldistance=1.05 # Jarak label dari pusat pie
    )
    
    ax.set_title('Sentiment Distribution | Posts')
    st.pyplot(fig)
    df['created_time'] = pd.to_datetime(df['created_time'])
    min_date = df['created_time'].min().date()
    max_date = df['created_time'].max().date()

    # Slider untuk memilih tanggal
    selected_date = st.slider("Select a date", min_value=min_date, max_value=max_date, value=min_date, format="YYYY-MM-DD")

    # Filter dataset berdasarkan tanggal yang dipilih
    # Pastikan selected_date adalah tipe datetime.date
    filtered_df = df[df['created_time'].dt.date == selected_date]

    # Gabungkan teks dari kolom 'self_text'
    combined_text = " ".join(filtered_df['self_text'].dropna())

    # Generate WordCloud
    if combined_text.strip():  # Pastikan teks tidak kosong
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
        
        # Tampilkan WordCloud
        st.subheader(f"WordCloud for {selected_date}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("No text available for the selected date!")

def check_reddit_sentiment():
    st.title("Check Your Reddit Sentiment")
    st.text("Input your Reddit post or comment to analyze its sentiment.")
    st.markdown("""
    <style>
    .stTextArea textarea {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    # Input teks dari pengguna
    user_input = st.text_area("Enter your Reddit text here:",)
    if st.button("Submit"):
        if user_input:
            # Identifikasi sentimen menggunakan TextBlob
            sentiment_score = TextBlob(user_input).sentiment.polarity
            if sentiment_score > 0:
                sentiment = "Positive"
            elif sentiment_score < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            # Tampilkan hasil sentimen
            st.subheader("Sentiment Result")
            st.write(f"The sentiment of your text is **{sentiment}**.")
            st.write(f"Sentiment Score: {sentiment_score:.2f}")
        else:
            st.warning("Please enter some text before submitting.")

# Fungsi utama
def main():
    df = load_data()

    # Preprocessing
    start_date = pd.to_datetime('2023-10-07')
    df = df.dropna(subset=['created_time', 'post_created_time'])
    df = df[(df['post_created_time'] >= start_date) & (df['created_time'] >= start_date)]
    df.isnull().sum().sum()
    #df = df.drop(columns=['ups', 'post_thumbs_ups', 'downs','post_total_awards_received',
                                        #'user_awardee_karma', 'user_awarder_karma','user_link_karma', 'user_comment_karma'])
    #df = df.iloc[:10000]
    if 'post_title' in df.columns:
        df=df[(~df['post_title'].isna())]
        df = df.drop_duplicates(subset=['post_title'])
        df['post_sentiment'] = df['post_title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        # Assign sentiment category based on sentiment score
        df['post_sentiment_category'] = df['post_sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    else:
        st.warning("Column 'post_title' is missing. Sentiment analysis will not be available.")

    # Sidebar menu
    with st.sidebar:
        app = option_menu(
            menu_title="All-about-Palestine",
            options=["Home", "History","Sentiment Analysis"],
            icons=["house", "clock-history", "graph-up-arrow"],
            styles={
                "container": {"padding": "5!important"},
                "icon": {"color": "orange"},
                "nav-link": {"font-size": "14px"},
            }
        )

    # Navigation
    if app == "Home":
        show_home()
    elif app == "History":
        show_history()
    elif app == "Sentiment Analysis":
        with st.sidebar:
            sentiment_menu = option_menu(
                menu_title="Choose an option:", 
                options=["Analysis Data", "Check Your Reddit"],
                styles={
                    "container": {"padding": "5!important"},
                    "icon": {"color": "orange"},
                    "nav-link": {"font-size": "14px"}},
            )
        if sentiment_menu == "Analysis Data":
            show_sentiment_analysis(df)
        elif sentiment_menu == "Check Your Reddit":
            check_reddit_sentiment()

if __name__ == "__main__":
    main()
