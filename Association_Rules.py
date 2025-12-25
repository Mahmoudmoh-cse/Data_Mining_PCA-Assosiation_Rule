import streamlit as st
import pandas as pd
import re
from itertools import combinations
from collections import defaultdict

st.set_page_config(page_title="Association Rules", page_icon="🔗", layout="wide")

# ------------------ Custom CSS
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}
.stButton > button {
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    color: #020617;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    border: none;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 12px 35px rgba(167,139,250,0.6);
}
h2 {
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}
[data-testid="stDataFrameContainer"] {
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header
st.markdown("<div class='card'><h2>🔗 Association Rule Mining</h2></div>", unsafe_allow_html=True)
st.markdown("""
Upload your transactional CSV to generate **frequent itemsets** and **association rules** dynamically.
Rules are calculated based on:

- **Support**: frequency of itemset in all transactions.
- **Confidence**: how often consequent appears when antecedent appears.
""")

# ------------------ Upload CSV
uploaded_file = st.file_uploader("Upload Transactional CSV", type=["csv"])

# ------------------ Functions
def clean_items(items):
    items = re.split(r'[;,|]', str(items))
    items = [i.strip().lower() for i in items if i.strip()]
    return sorted(set(items))

def get_frequent_itemsets(transactions, min_support=0.05, max_itemset_size=3):
    counts = defaultdict(int)
    num_transactions = len(transactions)
    for t in transactions:
        for k in range(1, min(len(t)+1, max_itemset_size+1)):
            for comb in combinations(t, k):
                counts[frozenset(comb)] += 1
    return {itemset: count/num_transactions for itemset, count in counts.items() if count/num_transactions >= min_support}

def generate_rules(freq_itemsets, min_confidence=0.4):
    rules = []
    for itemset, sup_ab in freq_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    if antecedent in freq_itemsets:
                        sup_a = freq_itemsets[antecedent]
                        confidence = sup_ab / sup_a
                        lift = sup_ab / (sup_a * freq_itemsets.get(consequent, 1))
                        if confidence >= min_confidence:
                            rules.append({
                                "antecedent": sorted(antecedent),
                                "consequent": sorted(consequent),
                                "support": round(sup_ab, 3),
                                "confidence": round(confidence, 3),
                                "lift": round(lift, 3)
                            })
    return rules

# ------------------ Process Uploaded CSV
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Clean dataset
    df.replace(['NULL','null','NaN',''], pd.NA, inplace=True)
    df.dropna(subset=['Transaction ID','Items Purchased'], inplace=True)
    df.drop_duplicates(subset=['Transaction ID','Items Purchased'], inplace=True)
    df['clean_items'] = df['Items Purchased'].apply(clean_items)
    df = df[df['clean_items'].str.len() > 0]

    transactions = df['clean_items'].tolist()

    # ------------------ Sliders
    min_support = st.slider("Minimum Support", 0.01, 1.0, 0.05, step=0.01)
    min_confidence = st.slider("Minimum Confidence", 0.01, 1.0, 0.4, step=0.01)

    # ------------------ Buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_raw = st.button("👀 Show Raw Dataset")
    with col2:
        show_cleaned = st.button("🧹 Show Cleaned Dataset")
    with col3:
        show_itemsets = st.button("📦 Show Frequent Itemsets")
    with col4:
        show_rules = st.button("📊 Show Association Rules")

    # ------------------ Show Tables
    if show_raw:
        st.dataframe(df.head())
    if show_cleaned:
        st.dataframe(df[['Transaction ID','clean_items']].head())

    # Frequent Itemsets
    freq_itemsets = get_frequent_itemsets(transactions, min_support)
    if show_itemsets:
        freq_df = pd.DataFrame([{"Itemset": ', '.join(list(k)), "Support": v} for k,v in freq_itemsets.items()])
        freq_df = freq_df.sort_values(by="Support", ascending=False)
        st.dataframe(freq_df)

    # Association Rules
    rules = generate_rules(freq_itemsets, min_confidence)
    if show_rules:
        if rules:
            rules_df = pd.DataFrame([{
                "Antecedent": ', '.join(r["antecedent"]),
                "Consequent": ', '.join(r["consequent"]),
                "Support": r["support"],
                "Confidence": r["confidence"],
                "Lift": r["lift"]
            } for r in rules])
            rules_df = rules_df.sort_values(by=["Confidence","Lift"], ascending=False)
            st.dataframe(rules_df)
        else:
            st.warning("No rules found with current Support & Confidence thresholds.")
