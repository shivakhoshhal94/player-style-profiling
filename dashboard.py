import pandas as pd
import streamlit as st
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("position_recommendations.csv")

df = load_data()

# Sidebar filters
st.sidebar.title("🔍 Player Filter")
selected_player = st.sidebar.selectbox("Select a player", sorted(df['Player'].unique()))

# Player-specific data
player_data = df[df['Player'] == selected_player].iloc[0]
st.title("⚽ Player Position Recommendation Dashboard")

st.subheader(f"📌 Player: {selected_player}")
st.write(f"**Actual Position(s):** {player_data['Pos']}")
st.write(f"**Top 1 Prediction:** {player_data['Top1_Pred']}")
st.write(f"**Top 2 Prediction:** {player_data['Top2_Pred']}")
st.write(f"**Top 3 Prediction:** {player_data['Top3_Pred']}")
st.success(f"Assessment: {player_data['Assessment']}")

# Position fit pie chart
st.subheader("📊 Assessment Summary")
summary_counts = df['Assessment'].value_counts()
fig_pie = px.pie(
    names=summary_counts.index,
    values=summary_counts.values,
    title="Assessment Summary of Player Position Fit",
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig_pie)

# Top-3 predicted position frequency
st.subheader("📈 Top-3 Predicted Positions Frequency")
top3_flat = pd.concat([
    df[['Top1_Pred']].rename(columns={'Top1_Pred': 'Position'}),
    df[['Top2_Pred']].rename(columns={'Top2_Pred': 'Position'}),
    df[['Top3_Pred']].rename(columns={'Top3_Pred': 'Position'})
])
top3_counts = top3_flat['Position'].value_counts().reset_index()
top3_counts.columns = ['Position', 'Frequency']
fig_bar = px.bar(top3_counts, x='Position', y='Frequency', 
                 title='Top-3 Predicted Positions (All Players)',
                 color='Position')
st.plotly_chart(fig_bar)

# Raw data view
with st.expander("📄 Show Full Dataset"):
    st.dataframe(df)
