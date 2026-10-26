import io
import json
import re
import hashlib
from datetime import datetime, date, time
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SmartSegment AI", page_icon="🎯", layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {
  --bg: #080d1b; --panel: #111a30; --panel2: #17233e;
  --line: rgba(160,174,214,.18); --text: #f4f7ff; --muted: #a9b6d5;
  --purple: #8b5cf6; --cyan: #67e8f9; --mint: #6ee7b7;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background:
    radial-gradient(ellipse at 5% 0%, rgba(124,58,237,.24), transparent 35%),
    radial-gradient(ellipse at 98% 8%, rgba(6,182,212,.16), transparent 29%),
    linear-gradient(135deg,#080d1b 0%,#0b1225 55%,#10182d 100%);
  color: var(--text);
}
[data-testid="stHeader"] { background: rgba(8,13,27,.78); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#111a31 0%,#0b1223 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: #e7edff; }
.block-container { max-width: 1500px; padding-top: 1.35rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family: 'Manrope',sans-serif !important; color: #f7f9ff !important; letter-spacing: -.035em; }
h1 { font-weight: 800 !important; }
h2,h3 { font-weight: 700 !important; }
p, label, [data-testid="stCaptionContainer"] { color: #d0d9f0; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg,rgba(28,41,72,.96),rgba(15,24,45,.96));
  border: 1px solid var(--line); border-radius: 17px; padding: 17px 19px;
  box-shadow: 0 12px 28px rgba(0,0,0,.15);
}
[data-testid="stMetricLabel"] { color: #aebce0 !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #fff !important; font-family: 'Manrope',sans-serif; font-weight: 800; }
.stButton > button, .stDownloadButton > button {
  border: 1px solid rgba(167,139,250,.4); border-radius: 12px;
  color: white; font-weight: 700; background: linear-gradient(100deg,#7c3aed,#4f46e5);
  box-shadow: 0 8px 22px rgba(79,70,229,.22); transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); border-color: #a78bfa; color: white;
  box-shadow: 0 12px 28px rgba(124,58,237,.32);
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-testid="stFileUploader"] section {
  background: rgba(15,24,45,.95) !important; color: #f8faff !important;
  border-color: var(--line) !important; border-radius: 12px !important;
}
[data-testid="stFileUploader"] {
  background: rgba(20,30,54,.65); border: 1px dashed rgba(167,139,250,.6);
  border-radius: 16px; padding: 10px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--line); border-radius: 13px; overflow: hidden;
}
[data-testid="stExpander"] {
  background: rgba(19,29,51,.66); border: 1px solid var(--line); border-radius: 14px;
}
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
hr { border-color: rgba(160,174,214,.16); }
.ss-hero {
  position: relative; overflow: hidden; border-radius: 24px; padding: 29px 31px;
  margin: 0 0 20px; border: 1px solid rgba(167,139,250,.28);
  background:
    radial-gradient(circle at 88% 12%,rgba(103,232,249,.20),transparent 26%),
    radial-gradient(circle at 67% 100%,rgba(167,139,250,.22),transparent 37%),
    linear-gradient(112deg,rgba(42,32,87,.98),rgba(18,31,60,.98) 60%,rgba(12,43,59,.98));
  box-shadow: 0 18px 48px rgba(0,0,0,.22);
}
.ss-hero .eyebrow {
  display: inline-block; padding: 6px 11px; border-radius: 999px;
  border: 1px solid rgba(103,232,249,.34); background: rgba(103,232,249,.08);
  color: #a5f3fc; font-size: 11px; font-weight: 800; letter-spacing: .13em;
  text-transform: uppercase;
}
.ss-hero h1 { font-size: clamp(2rem,4vw,3.2rem); line-height: 1.08; margin: 15px 0 9px; }
.ss-hero p { max-width: 820px; color: #d0d9f0; font-size: 1rem; line-height: 1.65; margin: 0; }
.ss-hero .symbol { float: right; font-size: 4.6rem; margin: 10px 22px 0 15px; color: #a5f3fc; text-shadow: 0 0 30px rgba(103,232,249,.45); }
.ss-kicker { color: #a5b4fc; font-size: .76rem; font-weight: 800; text-transform: uppercase; letter-spacing: .13em; margin: 1.2rem 0 .45rem; }
.ss-sidebrand { padding: 13px 12px; margin-bottom: 15px; border: 1px solid var(--line); border-radius: 15px; background: linear-gradient(135deg,rgba(124,58,237,.2),rgba(6,182,212,.08)); }
.ss-sidebrand strong { display:block; font-family:'Manrope',sans-serif; font-size:1.1rem; color:#fff; }
.ss-sidebrand span { font-size:.76rem; color:#a5f3fc; }
@media(max-width:700px) { .ss-hero { padding:22px 20px; } .ss-hero .symbol { display:none; } }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Helpers ---------------------------------
ALIASES = {
    "customer_id": ["customer_id", "customer id", "customerid", "client_id", "client id", "id"],
    "name": ["name", "customer_name", "customer name", "first_name", "first name"],
    "email": ["email", "email_address", "email address", "e-mail"],
    "phone": ["phone", "phone_number", "phone number", "mobile", "whatsapp"],
    "order_date": ["order_date", "order date", "purchase_date", "purchase date", "date"],
    "order_value": ["order_value", "order value", "amount", "order_amount", "order amount", "sales", "revenue", "total"],
    "order_id": ["order_id", "order id", "orderid", "transaction_id"],
    "last_purchase_date": ["last_purchase_date", "last purchase date", "last_order_date", "last order date"],
    "order_count": ["order_count", "order count", "orders", "purchase_frequency", "purchase frequency", "frequency"],
    "total_spend": ["total_spend", "total spend", "lifetime_value", "lifetime value", "customer_value", "total_sales"],
    "consent": ["consent", "marketing_consent", "marketing consent", "consent_flag", "consent flag", "opt_in", "opt in"],
    "returned": ["returned", "is_returned", "is returned", "return_flag", "return flag"],
    "complaint": ["complaint", "complained", "has_complaint", "has complaint"],
    "platform": ["platform", "preferred_platform", "preferred platform", "channel", "preferred_channel"],
    "active_hour": ["active_hour", "active hour", "active_hours", "active hours", "preferred_time", "preferred time"],
    "product_interest": ["product_interest", "product interest", "category", "product_category", "product category", "product"],
    "messages_last_7_days": ["messages_last_7_days", "messages last 7 days", "messages_this_week", "messages this week"],
    "unit_cost": ["unit_cost", "unit cost", "cost"],
    "margin_floor": ["margin_floor", "margin floor"],
}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def find_col(df, key):
    normed = {norm(c): c for c in df.columns}
    for alias in ALIASES.get(key, [key]):
        if norm(alias) in normed:
            return normed[norm(alias)]
    return None

def truthy(value, default=False):
    if pd.isna(value):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "consented", "opted in", "opt-in"}:
        return True
    if s in {"0", "false", "no", "n", "false", "not consented", "opted out", "opt-out"}:
        return False
    return default

def safe_str(value, default=""):
    if value is None or pd.isna(value):
        return default
    return str(value).strip()

def load_upload(upload):
    if upload.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            return pd.read_csv(upload, encoding="latin-1")
    return pd.read_excel(upload)

def customer_level_data(raw):
    """Return one row per customer with RFM inputs, supporting customer or order-level uploads."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    id_col = find_col(df, "customer_id")
    if not id_col:
        # A stable row ID lets the app still work for one-row-per-customer files.
        df["customer_id_generated"] = [f"CUST-{i+1:05d}" for i in range(len(df))]
        id_col = "customer_id_generated"

    date_col = find_col(df, "order_date")
    last_col = find_col(df, "last_purchase_date")
    count_col = find_col(df, "order_count")
    spend_col = find_col(df, "total_spend")
    value_col = find_col(df, "order_value")
    order_id_col = find_col(df, "order_id")

    # Order-level data: aggregate multiple transactions per customer.
    if date_col and value_col and df[id_col].duplicated().any():
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        aggregations = {value_col: "sum"}
        if date_col:
            aggregations[date_col] = "max"
        if order_id_col:
            aggregations[order_id_col] = "nunique"
        else:
            aggregations[value_col] = ["sum", "count"]
        for key in ["name", "email", "phone", "consent", "returned", "complaint",
                    "platform", "active_hour", "product_interest", "messages_last_7_days"]:
            c = find_col(df, key)
            if c and c not in aggregations:
                aggregations[c] = "last"
        grouped = df.groupby(id_col, dropna=False).agg(aggregations)
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = ["_".join(str(x) for x in col if x) for col in grouped.columns]
        grouped = grouped.reset_index()
        # Normalize aggregate names created by pandas.
        mapping = {}
        for c in grouped.columns:
            nc = norm(c)
            if nc in {"order_value_sum", "order_value_sum_sum"}:
                mapping[c] = "total_spend"
            elif nc in {"order_value_count", "order_value_count_count"}:
                mapping[c] = "order_count"
            elif nc in {norm(date_col), f"{norm(date_col)}_max"}:
                mapping[c] = "last_purchase_date"
            elif order_id_col and nc in {norm(order_id_col), f"{norm(order_id_col)}_nunique"}:
                mapping[c] = "order_count"
        grouped = grouped.rename(columns=mapping)
        df = grouped
        id_col = find_col(df, "customer_id") or id_col

    # Create normalized RFM inputs from either pre-aggregated customer data or orders.
    id_col = find_col(df, "customer_id") or id_col
    date_col = find_col(df, "order_date")
    last_col = find_col(df, "last_purchase_date")
    count_col = find_col(df, "order_count")
    spend_col = find_col(df, "total_spend")
    value_col = find_col(df, "order_value")

    if spend_col:
        df["total_spend_calc"] = pd.to_numeric(df[spend_col], errors="coerce").fillna(0)
    elif value_col:
        df["total_spend_calc"] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    else:
        df["total_spend_calc"] = 0.0

    if count_col:
        df["order_count_calc"] = pd.to_numeric(df[count_col], errors="coerce").fillna(1).clip(lower=1)
    else:
        df["order_count_calc"] = 1

    date_source = last_col or date_col
    if date_source:
        parsed = pd.to_datetime(df[date_source], errors="coerce")
        df["last_purchase_calc"] = parsed
        today = pd.Timestamp(date.today())
        df["recency_days"] = (today - parsed).dt.days.clip(lower=0).fillna(999)
    else:
        df["last_purchase_calc"] = pd.NaT
        df["recency_days"] = 999

    df["total_spend_calc"] = df["total_spend_calc"].clip(lower=0)
    df["order_count_calc"] = df["order_count_calc"].clip(lower=1)
    df["frequency_score"] = df["order_count_calc"].rank(pct=True)
    df["spend_score"] = df["total_spend_calc"].rank(pct=True)
    df["recency_score"] = (1 - df["recency_days"].rank(pct=True))
    df["rfm_score"] = df["recency_score"] + df["frequency_score"] + df["spend_score"]

    # Explainable segment defaults; use K-Means when there are enough distinct customers.
    if len(df) >= 5 and df[["recency_days", "order_count_calc", "total_spend_calc"]].nunique().min() > 1:
        features = np.column_stack([
            np.log1p(df["recency_days"].clip(0, 3650)),
            np.log1p(df["order_count_calc"]),
            np.log1p(df["total_spend_calc"]),
        ])
        scaled = StandardScaler().fit_transform(features)
        k = min(5, max(2, len(df) // 3))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        df["cluster_id"] = labels
        cluster_summary = df.groupby("cluster_id").agg(
            recency=("recency_days", "mean"),
            frequency=("order_count_calc", "mean"),
            spend=("total_spend_calc", "mean")
        )
        # Map clusters by their behaviour rather than arbitrary cluster number.
        names = {}
        for cid, row in cluster_summary.iterrows():
            if row["recency"] > max(60, cluster_summary["recency"].median() * 1.35):
                names[cid] = "Dormant" if row["recency"] > 180 else "At Risk"
            elif row["frequency"] >= cluster_summary["frequency"].quantile(.70) and row["spend"] >= cluster_summary["spend"].median():
                names[cid] = "Champions"
            elif row["recency"] <= cluster_summary["recency"].median() and row["frequency"] >= cluster_summary["frequency"].median():
                names[cid] = "Loyal"
            else:
                names[cid] = "New"
        df["segment"] = df["cluster_id"].map(names)
    else:
        def rule_segment(r):
            if r["recency_days"] > 180:
                return "Dormant"
            if r["recency_days"] > 60:
                return "At Risk"
            if r["order_count_calc"] >= 5 and r["total_spend_calc"] >= df["total_spend_calc"].median():
                return "Champions"
            if r["order_count_calc"] >= 2:
                return "Loyal"
            return "New"
        df["segment"] = df.apply(rule_segment, axis=1)

    # Fill missing optional customer attributes safely.
    for key in ["name", "email", "phone", "consent", "returned", "complaint",
                "platform", "active_hour", "product_interest", "messages_last_7_days"]:
        c = find_col(df, key)
        df[f"_{key}"] = df[c] if c else None
    df["_customer_id"] = df[id_col].astype(str)
    return df

def recommended_offer(segment):
    offers = {
        "Champions": (0, "VIP early access; no discount needed"),
        "Loyal": (5, "Bundle or new-arrivals offer"),
        "At Risk": (12, "Win-back offer"),
        "New": (5, "Welcome / second-purchase nudge"),
        "Dormant": (3, "Low-cost reactivation email"),
    }
    return offers.get(segment, (0, "Personalized product recommendation"))

def frequency_discount_band(row, max_discount):
    """
    Business guardrail: frequent purchasers receive smaller discounts;
    infrequent or lapsed purchasers may receive larger win-back offers.
    Returns (minimum, maximum, explanation).
    """
    try:
        frequency = max(1, int(float(row.get("order_count_calc", 1) or 1)))
    except (TypeError, ValueError):
        frequency = 1

    segment = safe_str(row.get("segment"), "New")
    try:
        recency = int(float(row.get("recency_days", 999) or 999))
    except (TypeError, ValueError):
        recency = 999

    if frequency >= 8:
        low, high, reason = 0, 2, "Very frequent buyer: reward with perks and keep discount low"
    elif frequency >= 5:
        low, high, reason = 2, 4, "Frequent buyer: small loyalty discount"
    elif frequency >= 3:
        low, high, reason = 4, 7, "Regular buyer: moderate repeat-purchase discount"
    elif frequency == 2:
        low, high, reason = 7, 10, "Two purchases: encourage the next order"
    else:
        low, high, reason = 8, 12, "One recorded purchase: encourage a second order"

    if segment == "At Risk" or recency > 90:
        low += 1
        high += 2
        reason += "; increased slightly because the customer is inactive"
    if segment == "Dormant" or recency > 180:
        low += 1
        high += 2
        reason += "; win-back adjustment for a long gap since purchase"

    high = min(int(max_discount), high)
    low = min(low, high)
    return max(0, low), max(0, high), reason


def ai_discount_recommendation(row, model, base_url, use_ollama, max_discount):
    """
    Ollama proposes an individual discount. A frequency-based business policy
    then constrains the final recommendation so high-frequency buyers receive
    lower discounts than customers who buy infrequently or have lapsed.
    """
    low, high, band_reason = frequency_discount_band(row, max_discount)
    segment = safe_str(row.get("segment"), "Unknown")
    recency = int(float(row.get("recency_days", 999) or 999))
    frequency = float(row.get("order_count_calc", 1) or 1)
    spend = float(row.get("total_spend_calc", 0) or 0)
    product = safe_str(row.get("_product_interest"), "unknown")
    platform = safe_str(row.get("_platform"), "unknown")

    fallback = int(round((low + high) / 2))
    if not use_ollama:
        return fallback, f"{band_reason}. Rules-based recommendation: {fallback}%.", "Frequency policy"

    prompt = f"""You are a careful discount decision agent for a direct-to-consumer business.
Recommend an integer discount percentage for this customer.
Use purchase frequency as a key factor: customers who buy more frequently should generally receive a smaller discount; customers who buy infrequently or have not purchased recently may receive a larger win-back discount.
The final allowed range for this specific customer is {low}% to {high}%. You MUST choose a number inside that range.
Do not use sensitive personal attributes. Do not invent data.
Return ONLY valid JSON:
{{"discount_pct": {fallback}, "reason": "brief reason"}}

Customer facts:
- Segment: {segment}
- Days since last purchase: {recency}
- Purchase/order frequency: {frequency}
- Total spend: {spend:.2f}
- Product interest: {product}
- Preferred channel: {platform}
Business rule: {band_reason}
"""
    try:
        response = ollama_generate(model, base_url, prompt)
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            raise ValueError("AI did not return JSON")
        result = json.loads(match.group(0))
        ai_pct = int(round(float(result.get("discount_pct", fallback))))
        # Hard constraint guarantees discount is within this customer's frequency tier.
        final_pct = max(low, min(high, ai_pct))
        reason = safe_str(result.get("reason"), "AI recommendation constrained by purchase-frequency policy")
        return final_pct, f"{reason}. Frequency policy: {band_reason}. Final discount: {final_pct}%.", "Ollama AI + frequency policy"
    except Exception as exc:
        return fallback, f"{band_reason}. Fallback used because Ollama could not recommend a valid value ({str(exc)[:80]}). Final discount: {fallback}%.", "Frequency policy fallback"


def make_template_message(row, discount, product):
    name = safe_str(row.get("_name"), "there")
    segment = safe_str(row.get("segment"), "New")
    product = product or safe_str(row.get("_product_interest"), "our latest collection")
    try:
        orders = max(1, int(float(row.get("order_count_calc", 1) or 1)))
    except (TypeError, ValueError):
        orders = 1
    try:
        days = int(float(row.get("recency_days", 999) or 999))
    except (TypeError, ValueError):
        days = 999

    discount_line = f" Your personalized offer is {int(discount)}% off." if int(discount) > 0 else " We have new arrivals and loyalty perks for you."
    if segment == "Champions" or orders >= 8:
        return f"Hi {name}, thank you for shopping with us {orders} times. As a valued regular customer, explore {product} and enjoy our latest loyalty perks. We appreciate your continued support: [link]"
    if segment == "Loyal" or orders >= 5:
        return f"Hi {name}, thanks for choosing us for your recent purchases. Based on your interest in {product}, we picked a few options you may like.{discount_line} Explore them here: [link]"
    if segment == "At Risk" or days > 60:
        return f"Hi {name}, it has been {days} days since your last recorded purchase. We would love to welcome you back to explore {product}.{discount_line} Take a look when you are ready: [link]"
    if segment == "Dormant" or days > 180:
        return f"Hi {name}, we have some new options in {product} since your last visit.{discount_line} If you are interested, explore the collection here: [link]"
    if orders <= 1:
        return f"Hi {name}, thanks for your first purchase! We thought you might like more from {product}.{discount_line} Discover more here: [link]"
    return f"Hi {name}, thanks for being our customer. We selected {product} based on your purchase activity.{discount_line} Explore the collection here: [link]"


def ollama_generate(model, base_url, prompt, timeout=90):
    url = base_url.rstrip("/") + "/api/generate"
    response = requests.post(url, json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.5}
    }, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


def generate_message(row, model, base_url, use_ollama, discount, product):
    fallback = make_template_message(row, discount, product)
    if not use_ollama:
        return fallback, "Behavior-based template"
    low, high, band_reason = frequency_discount_band(row, max(0, int(discount)))
    prompt = f"""You are a respectful direct-to-consumer marketing copywriter.
Write ONE distinct, natural, personalized message (35-75 words) for this specific customer.
Use the actual customer behavior below and mention it naturally when useful.
Do not invent facts, claims, product benefits, scarcity, or links. Use [link].
Do not mention sensitive personal traits. Never alter the approved discount.
If discount is 0%, do not claim there is a discount; emphasize relevant products or loyalty perks instead.
Name: {safe_str(row.get('_name'), 'there')}
Customer ID: {safe_str(row.get('_customer_id'), 'customer')}
Segment: {safe_str(row.get('segment'), 'New')}
Days since last purchase: {safe_str(row.get('recency_days'), 'unknown')}
Purchase frequency/order count: {safe_str(row.get('order_count_calc'), 'unknown')}
Total spend: {safe_str(row.get('total_spend_calc'), 'unknown')}
Product interest: {product or safe_str(row.get('_product_interest'), 'our products')}
Preferred channel: {safe_str(row.get('_platform'), 'Email')}
Approved discount: {int(discount)}%
Frequency policy: {band_reason}
Rules: produce a message that differs in wording and content according to this customer's behavior. Return only the message.
"""
    try:
        msg = ollama_generate(model, base_url, prompt).strip().strip('"')
        if len(msg) < 10:
            raise ValueError("Ollama returned an empty draft.")
        # Guard against the model inventing a different offer percentage.
        if int(discount) > 0 and re.search(r"\b\d{1,2}\s*%\b", msg):
            percentages = [int(x) for x in re.findall(r"\b(\d{1,2})\s*%\b", msg)]
            if any(pct != int(discount) for pct in percentages):
                msg = fallback
                return msg, "Behavior-based template (AI discount mismatch prevented)"
        return msg, "Ollama personalized message"
    except Exception as exc:
        return fallback, f"Behavior-based template fallback ({str(exc)[:80]})"


# ----------------------------- UI --------------------------------------
st.markdown("""
<div class="ss-hero">
  <div class="symbol">✦</div>
  <span class="eyebrow">AI-powered customer intelligence</span>
  <h1>SmartSegment AI</h1>
  <p>Understand every customer. Recommend smarter discounts. Create personalized campaigns — all in one intelligent workspace.</p>
</div>
<div class="ss-kicker">Customer intelligence&nbsp;&nbsp; / &nbsp;&nbsp;Campaign studio&nbsp;&nbsp; / &nbsp;&nbsp;Performance</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="ss-sidebrand">
      <strong>✦ SmartSegment AI</strong>
      <span>Customer growth workspace</span>
    </div>
    """, unsafe_allow_html=True)
    st.header("AI settings")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama model", value="llama3.2", help="Enter the model name shown by `ollama list`.")
    use_ollama = st.checkbox("Use local Ollama for message generation", value=True)
    st.divider()
    st.subheader("Campaign policy")
    campaign_budget = st.number_input("Campaign discount budget (₹)", min_value=0.0, value=10000.0, step=500.0)
    margin_floor = st.slider("Minimum gross margin after discount (%)", 0, 80, 20)
    max_discount = st.slider("Maximum discount (%)", 0, 30, 15)
    approval_threshold = st.slider("Human approval required above (%)", 1, 30, 15)
    max_messages_week = st.number_input("Max messages/customer/week", min_value=1, max_value=7, value=2)
    quiet_hours = st.checkbox("Respect quiet hours (10 PM–8 AM)", value=True)

st.markdown("### 01 · Upload customer data")
uploaded = st.file_uploader("Upload your customer Excel or CSV file", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a CSV or Excel file to start. For best results, include customer ID, order date or last purchase date, order count or order records, and total spend or order value.")
    with st.expander("Expected columns (the app also recognizes common alternatives)"):
        st.markdown("""
        - **Required for best analysis:** `customer_id` plus `order_date` and `order_value` for transaction-level data, OR `order_count`, `total_spend`, and `last_purchase_date` for customer-level data.
        - **Recommended safety fields:** `consent` / `marketing_consent`, `returned`, `complaint`, `messages_last_7_days`.
        - **Optional personalization:** `name`, `email`, `phone`, `platform`, `active_hour`, `product_interest`.
        - If consent is missing, the app marks customers **Hold — consent unknown**. It does not send messages.
        """)
    st.stop()

try:
    raw_df = load_upload(uploaded)
    if raw_df.empty:
        st.error("The uploaded file has no rows.")
        st.stop()
    analyzed = customer_level_data(raw_df)
except Exception as exc:
    st.error(f"Could not read or analyze this file: {exc}")
    st.stop()

st.success(f"Loaded {len(raw_df):,} source rows and prepared {len(analyzed):,} customer records.")
with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)

# Offers and guardrails. Consent must be explicitly present and affirmative.
discounts, offer_reasons, statuses, reasons, messages, engines = [], [], [], [], [], []
spend_col = find_col(analyzed, "total_spend")
cost_col = find_col(analyzed, "unit_cost")
discount_reasons = []
discount_engines = []
for _, row in analyzed.iterrows():
    segment = row["segment"]
    proposed, offer_reason, discount_engine = ai_discount_recommendation(
        row, ollama_model, ollama_url, use_ollama, max_discount
    )
    # The AI chooses the individual percentage; this app enforces the hard maximum.
    approved_discount = max(0, min(int(max_discount), int(proposed)))

    consent_col = find_col(analyzed, "consent")
    consent_value = row.get("_consent")
    consent_present = consent_col is not None
    consent_ok = truthy(consent_value, default=False) if consent_present else False
    returned_ok = not truthy(row.get("_returned"), default=False)
    complaint_ok = not truthy(row.get("_complaint"), default=False)
    messages_count = pd.to_numeric(pd.Series([row.get("_messages_last_7_days")]), errors="coerce").iloc[0]
    messages_count = 0 if pd.isna(messages_count) else int(messages_count)

    status, why = "Draft eligible", "Passed basic campaign checks"
    if not consent_ok:
        status, why = "Hold — consent missing/false", "Marketing consent must be explicitly true"
    elif not returned_ok:
        status, why = "Suppressed", "Returned order flagged"
    elif not complaint_ok:
        status, why = "Suppressed", "Customer complaint flagged"
    elif messages_count >= max_messages_week:
        status, why = "Hold — contact limit", "Weekly contact limit reached"
    elif approved_discount > approval_threshold:
        status, why = "Needs human approval", f"Discount exceeds approval threshold ({approval_threshold}%)"

    discounts.append(approved_discount)
    offer_reasons.append(offer_reason)
    discount_reasons.append(offer_reason)
    discount_engines.append(discount_engine)
    statuses.append(status)
    reasons.append(why)
    messages.append("")
    engines.append("")

analyzed["recommended_discount_pct"] = discounts
analyzed["offer_strategy"] = offer_reasons
analyzed["discount_decision_reason"] = discount_reasons
analyzed["discount_decision_engine"] = discount_engines
analyzed["campaign_status"] = statuses
analyzed["guardrail_reason"] = reasons

# Campaign budget cap: reduce discounts for lower-priority offers first.
estimated_discount_cost = (analyzed["total_spend_calc"] * analyzed["recommended_discount_pct"] / 100).sum()
if estimated_discount_cost > campaign_budget and estimated_discount_cost > 0:
    priority = {"Champions": 0, "Loyal": 1, "New": 2, "At Risk": 3, "Dormant": 4}
    order = analyzed.assign(_priority=analyzed["segment"].map(priority).fillna(5)).sort_values("_priority", ascending=False).index
    for idx in order:
        if estimated_discount_cost <= campaign_budget:
            break
        current = float(analyzed.at[idx, "recommended_discount_pct"])
        if current <= 0:
            continue
        saving = analyzed.at[idx, "total_spend_calc"] * current / 100
        analyzed.at[idx, "recommended_discount_pct"] = 0
        analyzed.at[idx, "guardrail_reason"] += "; discount removed to fit campaign budget"
        estimated_discount_cost -= saving

st.markdown("### 02 · Campaign overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(analyzed):,}")
c2.metric("Segments", f"{analyzed['segment'].nunique():,}")
c3.metric("Draft-eligible", int((analyzed["campaign_status"] == "Draft eligible").sum()))
c4.metric("Held / suppressed", int((analyzed["campaign_status"] != "Draft eligible").sum()))
left, right = st.columns([1, 1])
with left:
    st.markdown("**Customer segments**")
    st.bar_chart(analyzed["segment"].value_counts())
with right:
    st.markdown("**Guardrail status**")
    st.bar_chart(analyzed["campaign_status"].value_counts())

st.info(
    "Discounts are now constrained by purchase frequency: customers with 8+ orders receive 0–2%; "
    "5–7 orders receive 2–4%; 3–4 orders receive 4–7%; 2 orders receive 7–10%; "
    "1 order receives 8–12%, with a small extra allowance for inactive/at-risk customers. "
    "The configured maximum discount always applies."
)

st.markdown("### 03 · Customer intelligence")
display_cols = ["_customer_id", "segment", "recency_days", "order_count_calc", "total_spend_calc",
                "recommended_discount_pct", "campaign_status", "guardrail_reason"]
st.dataframe(analyzed[display_cols].rename(columns={
    "_customer_id": "customer_id", "recency_days": "recency_days",
    "order_count_calc": "order_count", "total_spend_calc": "total_spend"
}), use_container_width=True, hide_index=True)

st.markdown("### 04 · Personalized campaign studio")
product_override = st.text_input("Campaign/product focus (optional)", placeholder="e.g., new arrivals, skincare bundle, aquarium accessories")
eligible_count = int((analyzed["campaign_status"] == "Draft eligible").sum())
all_customer_count = len(analyzed)
st.write(
    f"Customers in file: **{all_customer_count}** · "
    f"Eligible for campaign drafts: **{eligible_count}** · "
    f"Held/suppressed: **{all_customer_count - eligible_count}**"
)
st.caption(
    "The AI can create an individual draft for every customer record. Customers without confirmed marketing consent, "
    "or customers flagged for complaints/returns/contact limits, will be marked as held and their messages will not be campaign-ready."
)
if use_ollama:
    st.caption(f"Local AI endpoint: {ollama_url} · Model: {ollama_model}. If Ollama is unavailable, safe template drafts are used.")
else:
    st.caption("Template mode is selected; drafts will use personalized rules-based templates rather than Ollama.")

generation_mode = st.radio(
    "Who should receive a draft?",
    ["All customers (including held records, clearly marked)", "Only campaign-eligible customers"],
    index=0,
    help="Choosing all customers creates review drafts for the full dataset. It does not override consent or safety holds."
)

if st.button("✨ Automatically generate each customer's message", type="primary"):
    progress = st.progress(0)
    status_text = st.empty()
    new_messages = []
    new_engines = []
    if generation_mode.startswith("All customers"):
        draft_indices = analyzed.index.tolist()
    else:
        draft_indices = analyzed.index[analyzed["campaign_status"] == "Draft eligible"].tolist()

    analyzed["personalized_message"] = ""
    analyzed["message_engine"] = ""
    analyzed["message_ready_status"] = ""
    total = len(draft_indices)
    for i, idx in enumerate(draft_indices):
        row = analyzed.loc[idx]
        msg, engine = generate_message(
            row, ollama_model, ollama_url, use_ollama,
            int(row["recommended_discount_pct"]), product_override
        )
        analyzed.at[idx, "personalized_message"] = msg
        analyzed.at[idx, "message_engine"] = engine
        if safe_str(row.get("campaign_status")) == "Draft eligible":
            analyzed.at[idx, "message_ready_status"] = "Ready for review"
        else:
            analyzed.at[idx, "message_ready_status"] = "HOLD — do not send: " + safe_str(row.get("guardrail_reason"), "policy check failed")
        progress.progress((i + 1) / max(1, total))
        status_text.write(f"Generating customer {i+1} of {total}…")
    st.session_state["analyzed_result"] = analyzed.copy()
    st.success(
        f"Generated {total} individual message drafts. "
        "These are drafts for review; no email or WhatsApp message has been sent."
    )
    st.rerun()

if "analyzed_result" in st.session_state:
    analyzed = st.session_state["analyzed_result"]
    st.markdown("### 05 · Message preview & export")
    message_col = "personalized_message"
    if message_col in analyzed.columns:
        preview = analyzed[analyzed[message_col].fillna("").astype(str).str.len() > 0]
        if not preview.empty:
            show_cols = [
                "_customer_id", "segment", "order_count_calc", "recency_days",
                "recommended_discount_pct", "discount_decision_reason",
                "campaign_status", "guardrail_reason"
            ]
            if "message_ready_status" in preview.columns:
                show_cols.append("message_ready_status")
            show_cols.extend([message_col, "message_engine"])
            show_cols = [c for c in show_cols if c in preview.columns]
            st.dataframe(
                preview[show_cols].rename(columns={"_customer_id": "customer_id"}),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "personalized_message": st.column_config.TextColumn("Full personalized message", width="large"),
                    "message_ready_status": st.column_config.TextColumn("Send status", width="large"),
                    "guardrail_reason": st.column_config.TextColumn("Why held / decision reason", width="large"),
                }
            )
            st.markdown("#### Read each customer's complete message")
            for _, message_row in preview.iterrows():
                customer_label = safe_str(message_row.get("_customer_id"), "Customer")
                segment_label = safe_str(message_row.get("segment"), "Unknown segment")
                discount_label = int(message_row.get("recommended_discount_pct", 0) or 0)
                status_label = safe_str(message_row.get("message_ready_status", message_row.get("campaign_status", "")))
                with st.expander(f"{customer_label} · {segment_label} · {discount_label}% discount · {status_label}"):
                    st.write(safe_str(message_row.get("personalized_message")))
                    st.caption(f"Discount decision: {safe_str(message_row.get('discount_decision_reason', ''))}")
        else:
            st.info("No message drafts yet. Click the generate button above.")
    st.download_button(
        "⬇️ Download full campaign results (CSV)",
        data=analyzed.to_csv(index=False).encode("utf-8-sig"),
        file_name="smartsegment_campaign_results.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Download message drafts only (CSV)",
        data=analyzed[analyzed.get("personalized_message", pd.Series("", index=analyzed.index)).fillna("").astype(str).str.len() > 0].to_csv(index=False).encode("utf-8-sig"),
        file_name="personalized_customer_messages.csv",
        mime="text/csv"
    )

with st.expander("How this matches your proposal"):
    st.markdown("""
    1. **Intake & validation:** reads CSV/Excel, normalizes common column names and creates one customer record.
    2. **Segmentation:** calculates recency, frequency and spend; applies K-Means when the data supports it, otherwise uses clear rules.
    3. **AI discount decision:** local Ollama recommends an individual discount percentage using each customer’s segment, recency, frequency, spend and product interest. The app enforces the configured maximum and campaign budget; review any flagged offers before use.
    4. **Guardrails:** requires explicit consent, suppresses returned/complaint records, respects weekly contact limits, and routes high discounts for human review.
    5. **Creative generation:** uses local Ollama for personalized drafts, with a template fallback.
    6. **Audit/export:** results and reasons are exported to CSV for review.

    **Important:** this version creates and exports drafts; it does not send real email/WhatsApp messages. Connectors, approval workflow, response tracking, contextual-bandit learning, and a true margin-aware optimizer are later implementation steps. Review the approval queue before any real campaign.
    """)
