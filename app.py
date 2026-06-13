import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Segoe UI Emoji'
from wordcloud import WordCloud
import seaborn as sns
import pickle

# Load ML model
model = pickle.load(open('sentiment_model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Page Title
st.sidebar.title("📱 WhatsApp Chat Analyzer")

# Upload File
uploaded_file = st.sidebar.file_uploader(
    "Choose a WhatsApp Chat File"
)

if uploaded_file is not None:

    # Read file
    bytes_data = uploaded_file.getvalue()

    data = bytes_data.decode("utf-8")

    # Preprocess
    df = preprocessor.preprocess(data)

    # Fetch unique users
    user_list = df['user'].unique().tolist()

    if 'group_notification' in user_list:
        user_list.remove('group_notification')

    user_list.sort()

    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox(
        "Show analysis for",
        user_list
    )

    # Button
    if st.sidebar.button("Show Analysis"):

        # Stats
        num_messages, words, num_media_messages, links = helper.fetch_stats(
            selected_user,
            df
        )

        st.title("📊 Top Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Messages")
            st.title(num_messages)

        with col2:
            st.header("Words")
            st.title(words)

        with col3:
            st.header("Media")
            st.title(num_media_messages)

        with col4:
            st.header("Links")
            st.title(links)

        # Monthly Timeline
        st.title("📅 Monthly Timeline")

        timeline = helper.monthly_timeline(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.plot(
            timeline['time'],
            timeline['message']
        )

        plt.xticks(rotation='vertical')

        st.pyplot(fig)

        # Daily Timeline
        st.title("📈 Daily Timeline")

        daily_timeline = helper.daily_timeline(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.plot(
            daily_timeline['only_date'],
            daily_timeline['message']
        )

        plt.xticks(rotation='vertical')

        st.pyplot(fig)

        # Busy Users
        if selected_user == 'Overall':

            st.title("👥 Most Busy Users")

            x, new_df = helper.most_busy_users(df)

            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values)

                plt.xticks(rotation='vertical')

                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("☁️ WordCloud")

        df_wc = helper.create_wordcloud(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.imshow(df_wc)

        st.pyplot(fig)

        # Most Common Words
        st.title("📝 Most Common Words")

        most_common_df = helper.most_common_words(
            selected_user,
            df
        )

        fig, ax = plt.subplots()

        ax.barh(
            most_common_df[0],
            most_common_df[1]
        )

        plt.xticks(rotation='vertical')

        st.pyplot(fig)

        # Emoji Analysis
        st.title("😀 Emoji Analysis")

        emoji_df = helper.emoji_helper(
            selected_user,
            df
        )

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:

            fig, ax = plt.subplots()

            ax.pie(
                emoji_df[1].head(),
                labels=emoji_df[0].head(),
                autopct="%0.2f"
            )

            st.pyplot(fig)

# ML Prediction Section
st.sidebar.title("🤖 ML Prediction")

input_sms = st.sidebar.text_area(
    "Enter Message"
)

if st.sidebar.button('Predict Spam'):

    # Transform text
    vector_input = cv.transform([input_sms])

    # Prediction
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("⚠️ Spam Message")

    else:
        st.success("✅ Normal Message")