import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="East Africa Financial Inclusion Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦",
)

# ---------------- Theme state ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ---------------- Theme palettes ----------------
LIGHT = {
    "bg":           "#FFFEFB",
    "bg_elev":      "#FFFFFF",
    "bg_soft":      "#F4FAFD",
    "bg_input":     "#FFFFFF",
    "text":         "#1A1A1A",
    "text_muted":   "#4A5568",
    "heading":      "#00668C",
    "accent":       "#71C4EF",
    "accent_hover": "#00668C",
    "border":       "#E2E8F0",
    "success_bg":   "#E0F7FA",
    "success_fg":   "#00668C",
    "success_bd":   "#00668C",
    "danger_bg":    "#FFEBEE",
    "danger_fg":    "#C62828",
    "danger_bd":    "#C62828",
    "shadow":       "0 2px 8px rgba(0, 102, 140, 0.08)",
    "shadow_hover": "0 4px 16px rgba(0, 102, 140, 0.16)",
    "chart_color":  "#00668C",
}

DARK = {
    "bg":           "#0E1117",
    "bg_elev":      "#1A1F2E",
    "bg_soft":      "#161B26",
    "bg_input":     "#0B1220",
    "text":         "#E8ECF1",
    "text_muted":   "#A0AEC0",
    "heading":      "#71C4EF",
    "accent":       "#00B4D8",
    "accent_hover": "#71C4EF",
    "border":       "#2D3748",
    "success_bg":   "rgba(46, 213, 115, 0.18)",
    "success_fg":   "#7BED9F",
    "success_bd":   "#2ED573",
    "danger_bg":    "rgba(255, 107, 107, 0.18)",
    "danger_fg":    "#FF7F7F",
    "danger_bd":    "#FF6B6B",
    "shadow":       "0 2px 8px rgba(0, 0, 0, 0.4)",
    "shadow_hover": "0 4px 16px rgba(0, 0, 0, 0.6)",
    "chart_color":  "#71C4EF",
}

P = DARK if st.session_state.theme == "Dark" else LIGHT

# ---------------- CSS ----------------
st.markdown(
    f"""
    <style>
    :root {{
        --bg:           {P['bg']};
        --bg-elev:      {P['bg_elev']};
        --bg-soft:      {P['bg_soft']};
        --bg-input:     {P['bg_input']};
        --text:         {P['text']};
        --text-muted:   {P['text_muted']};
        --heading:      {P['heading']};
        --accent:       {P['accent']};
        --accent-hover: {P['accent_hover']};
        --border:       {P['border']};
        --success-bg:   {P['success_bg']};
        --success-fg:   {P['success_fg']};
        --success-bd:   {P['success_bd']};
        --danger-bg:    {P['danger_bg']};
        --danger-fg:    {P['danger_fg']};
        --danger-bd:    {P['danger_bd']};
        --shadow:       {P['shadow']};
        --shadow-hover: {P['shadow_hover']};
    }}

    /* App background */
    .stApp {{
        background-color: var(--bg);
        color: var(--text);
    }}

    /* Top header (the white nav bar in your screenshot) */
    header[data-testid="stHeader"] {{
        background-color: var(--bg) !important;
        border-bottom: 1px solid var(--border);
    }}
    header[data-testid="stHeader"] * {{
        color: var(--text) !important;
    }}
    /* Toolbar / Deploy button area */
    div[data-testid="stToolbar"] {{
        background-color: transparent !important;
    }}
    div[data-testid="stToolbar"] button,
    div[data-testid="stToolbar"] svg {{
        color: var(--text) !important;
        fill: var(--text) !important;
    }}
    /* Status decoration line at top */
    div[data-testid="stDecoration"] {{
        background: linear-gradient(90deg, var(--accent), var(--heading)) !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-soft);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {{
        color: var(--text);
    }}

    /* Sidebar collapse / expand arrows */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapsedControl"],
    button[kind="header"] {{
        background-color: var(--accent) !important;
        border-radius: 8px !important;
        opacity: 1 !important;
    }}
    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="stSidebarCollapsedControl"] svg,
    button[kind="header"] svg {{
        color: white !important;
        fill: white !important;
        width: 22px !important;
        height: 22px !important;
    }}
    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="stSidebarCollapsedControl"]:hover,
    button[kind="header"]:hover {{
        background-color: var(--accent-hover) !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--heading) !important;
        font-weight: 700 !important;
    }}
    h1 {{ letter-spacing: -0.5px; }}

    /* Body text in main area */
    .main p, .main label, .main li {{
        color: var(--text);
    }}

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: var(--bg-input) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }}
    .stNumberInput > div > div {{
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }}
    .stNumberInput button {{
        background-color: var(--bg-input) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }}
    .stNumberInput button svg {{
        fill: var(--text) !important;
        color: var(--text) !important;
    }}
    /* Selectbox dropdown caret + value text */
    .stSelectbox div[data-baseweb="select"] svg {{
        fill: var(--text) !important;
    }}
    .stSelectbox div[data-baseweb="select"] span {{
        color: var(--text) !important;
    }}
    /* Selectbox popover menu */
    div[data-baseweb="popover"] ul {{
        background-color: var(--bg-elev) !important;
        border: 1px solid var(--border) !important;
    }}
    div[data-baseweb="popover"] li {{
        background-color: var(--bg-elev) !important;
        color: var(--text) !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: var(--bg-soft) !important;
    }}
    /* Input labels */
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label {{
        color: var(--text) !important;
        font-weight: 600 !important;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, var(--accent), var(--accent-hover));
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1.05em;
        width: 100%;
        box-shadow: var(--shadow);
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
        filter: brightness(1.05);
    }}
    .stButton>button p {{
        color: white !important;
    }}

    /* Bordered containers (cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--bg-elev) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: var(--shadow);
        transition: box-shadow 0.2s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: var(--shadow-hover);
    }}

    /* Hero */
    .hero {{
        background: linear-gradient(135deg, var(--bg-soft), var(--bg-elev));
        border: 1px solid var(--border);
        border-left: 5px solid var(--accent);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: var(--shadow);
    }}
    .hero p {{
        color: var(--text-muted) !important;
        font-size: 1.05em;
        margin: 0;
        line-height: 1.6;
    }}
    .hero h3 {{ margin-top: 0; }}

    /* Result boxes */
    .success-box {{
        background-color: var(--success-bg);
        border-left: 5px solid var(--success-bd);
        padding: 1.25rem 1.5rem;
        color: var(--success-fg) !important;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.15em;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }}
    .success-box strong {{ color: var(--success-fg) !important; }}

    .predict-box {{
        background-color: var(--danger-bg);
        border-left: 5px solid var(--danger-bd);
        padding: 1.25rem 1.5rem;
        color: var(--danger-fg) !important;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.15em;
        margin: 1rem 0;
        box-shadow: var(--shadow);
    }}
    .predict-box strong {{ color: var(--danger-fg) !important; }}

    /* Metric tiles */
    div[data-testid="stMetric"] {{
        background-color: var(--bg-elev);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: var(--shadow);
    }}
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {{
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        color: var(--heading) !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}

    /* Bar chart container */
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stArrowVegaLiteChart"],
    .element-container:has(.stVegaLiteChart) {{
        background-color: var(--bg-elev) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: var(--shadow);
    }}
    /* Force Vega chart canvas transparent so dark bg shows through */
    .vega-embed,
    .vega-embed canvas,
    .vega-embed svg {{
        background-color: transparent !important;
    }}
    .vega-embed .background {{
        fill: transparent !important;
    }}
    /* All vega text -> theme text color */
    .vega-embed text,
    .vega-embed .role-axis-label text,
    .vega-embed .role-axis-title text,
    .vega-embed .role-legend-label text,
    .vega-embed .role-legend-title text {{
        fill: var(--text) !important;
        font-weight: 600 !important;
    }}
    /* Axis lines */
    .vega-embed .role-axis line,
    .vega-embed .role-axis path,
    .vega-embed .role-axis-grid line {{
        stroke: var(--border) !important;
    }}

    /* Divider */
    hr {{
        border-color: var(--border) !important;
        margin: 2rem 0 !important;
    }}

    /* Sidebar badges */
    .badge {{
        display: inline-block;
        background-color: var(--accent);
        color: white !important;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: var(--text-muted) !important;
        font-size: 0.85em;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }}

    /* Radio (theme toggle) */
    div[role="radiogroup"] label {{
        color: var(--text) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🏦 About the Project")

    # Theme toggle
    st.markdown("#### 🎨 Appearance")
    theme_choice = st.radio(
        "Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        Predicts the likelihood of **financial inclusion** in East Africa
        using demographic and socio-economic factors.
        """
    )

    st.markdown("#### 📊 Dataset")
    st.markdown(
        """
        <span class="badge">FinScope 2018</span>
        <span class="badge">23,524 respondents</span>
        <span class="badge">4 countries</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🎯 Goal")
    st.write(
        "Identify key drivers of financial inclusion so stakeholders can "
        "target resources to bridge the digital divide and provide more "
        "accessible formal banking services."
    )

    st.markdown("---")
    st.caption("Built with Streamlit • Scikit-learn")

# ---------------- Header ----------------
st.title("East Africa Financial Inclusion Predictor")

st.markdown(
    """
    <div class="hero">
        <h3>How the model works</h3>
        <p>
            This predictive model evaluates an individual's demographic and
            socioeconomic profile to determine their likelihood of having a
            formal bank account. Fill in the details below and hit
            <b>Predict</b> to see the results.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Inputs ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### 👤 Demographics")
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=30, step=1,
            help="Respondent's age in years",
        )
        gender = st.selectbox(
            "Gender", options=["Female", "Male"],
            help="Respondent's gender",
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🎓 Socio-Economics")
        education = st.selectbox(
            "Education Level",
            options=[
                "No formal education / Other",
                "Primary education",
                "Secondary education",
                "Vocational/Specialised training",
                "Tertiary education",
            ],
            help="Highest level of education completed",
        )
        cellphone = st.selectbox(
            "Cellphone Access", options=["No", "Yes"],
            help="Does the respondent have access to a cellphone?",
        )

st.write("")

# ---------------- Model ----------------
@st.cache_resource
def load_model():
    return joblib.load("inclusion_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(
        f"Could not load the model. Ensure 'inclusion_model.pkl' is in the "
        f"current directory and compatible. Error: {e}"
    )
    model = None

# Center the predict button
btn_l, btn_c, btn_r = st.columns([1, 2, 1])
with btn_c:
    predict_clicked = st.button("🔮 Predict Financial Inclusion")

# ---------------- Prediction ----------------
if predict_clicked:
    if model is not None:
        gender_val = 1 if gender == "Male" else 0
        cell_val = 1 if cellphone == "Yes" else 0
        edu_map = {
            "No formal education / Other": 0,
            "Primary education": 1,
            "Secondary education": 2,
            "Vocational/Specialised training": 3,
            "Tertiary education": 4,
        }
        edu_val = edu_map[education]

        input_data = np.zeros((1, 31))
        scaled_age = (age - 38.8) / 16.5
        input_data[0, 1] = cell_val
        input_data[0, 3] = scaled_age
        input_data[0, 4] = gender_val
        input_data[0, 5] = edu_val

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.markdown("---")
            st.markdown("### 📈 Results")

            if prediction == 1:
                st.markdown(
                    '<div class="success-box">✅ The model predicts this '
                    'individual is <strong>Banked</strong>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="predict-box">⚠️ The model predicts this '
                    'individual is <strong>Unbanked</strong>.</div>',
                    unsafe_allow_html=True,
                )

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("Banked Probability", f"{probability[1]*100:.1f}%")
            m2.metric("Unbanked Probability", f"{probability[0]*100:.1f}%")
            m3.metric("Confidence", f"{max(probability)*100:.1f}%")

            st.markdown("#### Prediction Probability")
            prob_df = pd.DataFrame(
                {
                    "Status": ["Unbanked", "Banked"],
                    "Probability": [probability[0], probability[1]],
                }
            )

            axis_color = P["text"]
            grid_color = P["border"]
            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X(
                        "Status:N",
                        axis=alt.Axis(
                            labelColor=axis_color,
                            titleColor=axis_color,
                            labelFontSize=13,
                            labelFontWeight="bold",
                            domainColor=grid_color,
                            tickColor=grid_color,
                        ),
                    ),
                    y=alt.Y(
                        "Probability:Q",
                        axis=alt.Axis(
                            labelColor=axis_color,
                            titleColor=axis_color,
                            gridColor=grid_color,
                            domainColor=grid_color,
                            tickColor=grid_color,
                            format=".0%",
                        ),
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=["Unbanked", "Banked"],
                            range=[P["danger_bd"], P["chart_color"]],
                        ),
                        legend=None,
                    ),
                    tooltip=[
                        "Status",
                        alt.Tooltip("Probability:Q", format=".1%"),
                    ],
                )
                .properties(height=280, background="transparent")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)

        except ValueError as ve:
            st.error(f"Prediction Error: {ve}")
            st.info(
                "Note: The loaded model expects a different number of "
                "features. If the model was trained on the full 32-feature "
                "dataset, you will need to provide all 32 inputs or re-train "
                "the model on the 4 features used in this app."
            )

# ---------------- Footer ----------------
st.markdown(
    '<div class="footer">East Africa Financial Inclusion Predictor · '
    'Capstone Project</div>',
    unsafe_allow_html=True,
)
