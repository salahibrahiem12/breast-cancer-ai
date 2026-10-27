"""Streamlit research prototype with classification disabled by a safety gate."""

import io
from pathlib import Path

import streamlit as st

from src.config import FIGURES_DIR
from src.eligibility import (
    UNSUPPORTED_IMAGE_MESSAGE,
    UnsupportedImageError,
    decode_rgb_image,
    require_verified_ultrasound,
)


st.set_page_config(
    page_title="BreastCare AI Research Prototype",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --medical-blue: #1d2c43;
        --medical-pink: #e74491;
        --medical-purple: #6554ee;
        --medical-ink: #172943;
        --medical-muted: #61738d;
        --medical-surface: #ffffff;
        --medical-border: #e3eaf4;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 18%, rgba(251, 224, 242, 0.72), transparent 30rem),
            linear-gradient(180deg, #f8faff 0%, #ffffff 52%, #f6f9fd 100%);
        color: var(--medical-ink);
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        max-width: 1120px;
        padding: 0 1.5rem 2rem;
    }
    h1, h2, h3, h4, p {
        color: var(--medical-ink);
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0 -1.5rem;
        padding: 0.72rem 1.5rem;
        color: white;
        background: linear-gradient(105deg, #17253a, #1d304a);
        box-shadow: 0 4px 16px rgba(20, 39, 65, 0.16);
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 1.12rem;
        font-weight: 760;
        letter-spacing: -0.02em;
        color: #f7f9ff;
    }
    .brand-ribbon {
        color: #f05aa1;
        font-size: 1.9rem;
        line-height: 1;
    }
    .brand-name span {
        color: #ed64a9;
    }
    .brand-tagline {
        display: block;
        margin-top: 0.12rem;
        color: #bdc9da;
        font-size: 0.58rem;
        font-weight: 550;
        letter-spacing: 0.08em;
    }
    .ai-pill {
        border: 1px solid rgba(14, 202, 177, 0.32);
        border-radius: 999px;
        padding: 0.38rem 0.82rem;
        color: #3ce1c3;
        background: rgba(13, 184, 159, 0.11);
        font-size: 0.75rem;
        font-weight: 650;
    }
    .hero {
        display: grid;
        grid-template-columns: 1.08fr 0.92fr;
        align-items: center;
        gap: 1.2rem;
        padding: 2rem 0 1.25rem;
    }
    .hero-copy {
        max-width: 570px;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        margin-bottom: 0.7rem;
        padding: 0.33rem 0.76rem;
        border-radius: 999px;
        color: #d43d86;
        background: #fde8f3;
        font-size: 0.76rem;
        font-weight: 720;
    }
    .hero-title {
        margin: 0;
        color: #172943;
        font-size: clamp(2.15rem, 4vw, 3.05rem);
        font-weight: 820;
        letter-spacing: -0.055em;
        line-height: 1.04;
    }
    .hero-title .pink {
        color: #d74399;
        background: linear-gradient(95deg, #e74491, #7056e9);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-description {
        max-width: 500px;
        margin: 0.9rem 0 0.8rem;
        color: #536983;
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .hero-disclaimer {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        color: #71839b;
        font-size: 0.72rem;
        line-height: 1.45;
    }
    .hero-visual {
        position: relative;
        min-height: 240px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-glow {
        position: absolute;
        width: 250px;
        height: 210px;
        border-radius: 50%;
        background: radial-gradient(circle, #f6d5ee 0%, rgba(246, 213, 238, 0) 72%);
    }
    .ultrasound-frame {
        position: relative;
        z-index: 1;
        width: min(74%, 310px);
        padding: 0.42rem;
        overflow: hidden;
        border: 4px solid #fff;
        border-radius: 18px;
        background: #17253a;
        box-shadow: 0 14px 32px rgba(63, 76, 130, 0.22);
        transform: translate(-12px, 12px);
    }
    .ultrasound-frame img {
        display: block;
        width: 100%;
        height: 160px;
        border-radius: 10px;
        object-fit: cover;
        filter: saturate(0.65) contrast(1.05);
    }
    .scan-art {
        height: 160px;
        border-radius: 10px;
        background:
            radial-gradient(ellipse at 63% 55%, rgba(12, 17, 25, 0.9) 0 11%, transparent 12%),
            radial-gradient(ellipse at 60% 52%, rgba(228, 229, 222, 0.82) 0 1.5%, transparent 3%),
            repeating-linear-gradient(176deg, transparent 0 7px, rgba(226, 231, 231, 0.2) 8px 10px, transparent 11px 16px),
            linear-gradient(180deg, #9aa2a2 0%, #353d42 40%, #10171e 100%);
        filter: contrast(1.08);
    }
    .scan-marker {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 70px;
        height: 70px;
        border: 2px dashed #f44da6;
        border-radius: 50%;
        background: rgba(227, 66, 156, 0.28);
        transform: translate(-50%, -50%);
    }
    .analysis-float {
        position: absolute;
        z-index: 2;
        top: 6px;
        right: 3%;
        width: 152px;
        padding: 0.9rem 0.85rem;
        border: 1px solid #e8eafa;
        border-radius: 17px;
        background: rgba(255, 255, 255, 0.97);
        box-shadow: 0 12px 30px rgba(58, 65, 115, 0.16);
    }
    .analysis-title {
        margin-bottom: 0.55rem;
        color: #2865bc;
        font-size: 0.76rem;
        font-weight: 760;
    }
    .analysis-item {
        margin: 0.34rem 0;
        color: #40546e;
        font-size: 0.66rem;
        white-space: nowrap;
    }
    .analysis-item b {
        margin-right: 0.34rem;
        color: #0eaa75;
        font-size: 0.85rem;
    }
    .upload-card {
        margin: 0.35rem 0 1.15rem;
        padding: 0.8rem 0.85rem 0.65rem;
        border: 1px solid #e4eaf3;
        border-radius: 19px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 8px 30px rgba(32, 54, 91, 0.07);
    }
    .st-key-upload_card {
        margin: 0.35rem 0 1.15rem;
        padding: 0.8rem 0.85rem 0.65rem;
        border: 1px solid #e4eaf3;
        border-radius: 19px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 8px 30px rgba(32, 54, 91, 0.07);
    }
    .upload-heading {
        margin: 0.2rem 0 0.2rem;
        text-align: center;
        color: #203652;
        font-size: 1.15rem;
        font-weight: 760;
    }
    .upload-copy {
        margin: 0 0 0.7rem;
        text-align: center;
        color: #73839b;
        font-size: 0.8rem;
    }
    .upload-icon {
        display: block;
        margin: 0 auto 0.35rem;
        color: #7555f0;
        font-size: 2.3rem;
        line-height: 1;
        text-align: center;
    }
    div[data-testid="stFileUploader"] {
        border: 1.5px dashed #cbd9ed;
        border-radius: 13px;
        background: #fbfdff;
    }
    [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        min-height: 105px;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        padding: 0.8rem !important;
        border: 1.5px dashed #cbd9ed !important;
        border-radius: 12px;
        background: #fbfdff !important;
        color: #536983 !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #b6c8e3 !important;
        background: #f5f8ff !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #d6deed;
        border-radius: 8px;
        background: #ffffff !important;
        color: #2e4564 !important;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0.2rem 0.5rem;
    }
    div.stButton > button[kind="primary"] {
        min-height: 2.8rem;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(100deg, #ed478d, #6956eb);
        color: white;
        font-weight: 720;
        box-shadow: 0 5px 13px rgba(181, 73, 173, 0.2);
    }
    div.stButton > button[kind="primary"]:hover {
        border: 0;
        background: linear-gradient(100deg, #db347c, #5744d8);
        color: white;
    }
    .upload-note {
        margin-top: 0.75rem;
        padding: 0.62rem 0.8rem;
        border-radius: 9px;
        color: #3975bc;
        background: #eaf4ff;
        font-size: 0.73rem;
    }
    .feature-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin: 0.5rem 0 1.4rem;
    }
    .feature {
        display: flex;
        gap: 0.65rem;
        align-items: flex-start;
        min-height: 58px;
        padding: 0.3rem 0.55rem;
        border-right: 1px solid #e1e8f1;
    }
    .feature:last-child {
        border-right: 0;
    }
    .feature-icon {
        display: grid;
        width: 38px;
        height: 38px;
        flex: 0 0 38px;
        place-items: center;
        border-radius: 50%;
        color: #d94297;
        background: #fce6f3;
        font-size: 1rem;
        font-weight: 800;
    }
    .feature:nth-child(2) .feature-icon {
        color: #6b58e9;
        background: #eeebff;
    }
    .feature:nth-child(3) .feature-icon {
        color: #0aaf7d;
        background: #e1f8ef;
    }
    .feature:nth-child(4) .feature-icon {
        color: #3385e8;
        background: #e3f0ff;
    }
    .feature-title {
        margin: 0.16rem 0 0.17rem;
        color: #263d5a;
        font-size: 0.72rem;
        font-weight: 740;
    }
    .feature-copy {
        margin: 0;
        color: #71829a;
        font-size: 0.65rem;
        line-height: 1.35;
    }
    .diagnosis-card {
        margin: 0.8rem 0 0.3rem;
        padding: 1rem 1.2rem;
        border: 1px solid var(--medical-border);
        border-left: 6px solid #18a879;
        border-radius: 15px;
        background: #fff;
        box-shadow: 0 8px 24px rgba(32, 54, 91, 0.08);
    }
    .diagnosis-card.malignant {
        border-left-color: #d84361;
        background: #fff9fa;
    }
    .diagnosis-card h2 {
        margin: 0 0 0.2rem;
        font-size: 1.3rem;
    }
    .diagnosis-card p {
        margin: 0;
        color: var(--medical-muted);
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #e74491, #7056e9);
    }
    [data-testid="stImage"] img {
        border: 1px solid var(--medical-border);
        border-radius: 12px;
    }
    @media (max-width: 640px) {
        .hero {
            grid-template-columns: 1fr;
            padding-top: 1.35rem;
        }
        .hero-visual {
            min-height: 205px;
        }
        .ultrasound-frame img {
            height: 135px;
        }
        .analysis-float {
            right: 4%;
        }
        .feature-strip {
            grid-template-columns: repeat(2, 1fr);
        }
        .feature:nth-child(2) {
            border-right: 0;
        }
    }
    @media (max-width: 460px) {
        .block-container {
            padding-right: 0.9rem;
            padding-left: 0.9rem;
        }
        .topbar {
            margin-right: -0.9rem;
            margin-left: -0.9rem;
            padding-right: 0.9rem;
            padding-left: 0.9rem;
        }
        .ai-pill {
            font-size: 0.65rem;
            padding: 0.32rem 0.58rem;
        }
        .hero-title {
            font-size: 2.25rem;
        }
        .feature-strip {
            grid-template-columns: 1fr;
        }
        .feature {
            border-right: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <svg class="brand-ribbon" width="31" height="38" viewBox="0 0 32 40" aria-hidden="true">
              <path d="M11 3C4 3 3 11 8 17L20 36M21 3C28 3 29 11 24 17L12 36"
                fill="none" stroke="#ef59a2" stroke-width="5.5" stroke-linecap="round"/>
            </svg>
            <div class="brand-name">
              BreastCare <span>AI</span>
              <small class="brand-tagline">RESEARCH PROTOTYPE · NOT CLINICALLY VALIDATED</small>
            </div>
          </div>
          <div class="ai-pill">✦ &nbsp; Research prototype</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-copy">
            <div class="eyebrow">✦ &nbsp; Experimental prototype</div>
            <h1 class="hero-title">Breast Ultrasound <span style="color:#344eaa">AI</span><br>
              Research <span class="pink">Prototype</span>
            </h1>
            <p class="hero-description">
              An experimental research interface for breast ultrasound image
              classification. Classification is currently disabled for safety.
            </p>
            <div class="hero-disclaimer">
              <span aria-hidden="true">🛡</span>
              <span>Not clinically validated and not a replacement for clinical assessment.</span>
            </div>
          </div>
          <div class="hero-visual" aria-label="Illustrative ultrasound-style graphic, not patient data">
            <div class="hero-glow"></div>
            <div class="ultrasound-frame">
              <div class="scan-art" role="img" aria-label="Illustrative ultrasound scan"></div>
            </div>
            <div class="analysis-float">
              <div class="analysis-title">◉ &nbsp; Safety status</div>
              <div class="analysis-item"><b>!</b> Modality detector unavailable</div>
              <div class="analysis-item"><b>!</b> Classification disabled</div>
              <div class="analysis-item"><b>!</b> No diagnosis is produced</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_feature_strip() -> None:
    st.markdown(
        """
        <div class="feature-strip">
          <div class="feature">
            <div class="feature-icon">◉</div>
            <div><p class="feature-title">Research Prototype</p>
              <p class="feature-copy">Experimental image<br>classification research.</p></div>
          </div>
          <div class="feature">
            <div class="feature-icon">◉</div>
            <div><p class="feature-title">Visual Explanation</p>
              <p class="feature-copy">No explanation shown<br>without eligibility.</p></div>
          </div>
          <div class="feature">
            <div class="feature-icon">♧</div>
            <div><p class="feature-title">Safety First</p>
              <p class="feature-copy">Classification is<br>disabled for safety.</p></div>
          </div>
          <div class="feature">
            <div class="feature-icon">▤</div>
            <div><p class="feature-title">Educational Use</p>
              <p class="feature-copy">Not validated for<br>clinical use.</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_advanced_sections() -> None:
    """Keep analytics and architecture available without crowding the main flow."""
    with st.expander("Advanced", expanded=False):
        metrics_tab, architecture_tab = st.tabs(
            ["Historical Metrics (not clinical)", "System Architecture"]
        )

        with metrics_tab:
            st.caption(
                "Historical research charts only. They are not patient-independent "
                "validation and do not support clinical use."
            )
            chart_columns = st.columns(2)
            chart_paths = (
                (FIGURES_DIR / "confusion_matrix_breastnet.png", "Confusion matrix"),
                (FIGURES_DIR / "roc_curves_breastnet.png", "ROC curves"),
            )
            for column, (chart_path, caption) in zip(chart_columns, chart_paths):
                with column:
                    if chart_path.is_file():
                        st.image(str(chart_path), caption=caption, use_container_width=True)
                    else:
                        st.info(f"{caption} are not available yet.")

            learning_curves = FIGURES_DIR / "learning_curves_breastnet.png"
            if learning_curves.is_file():
                st.image(
                    str(learning_curves),
                    caption="Saved training and validation curves",
                    use_container_width=True,
                )

        with architecture_tab:
            st.markdown(
                """
                The experimental ResNet-18 was trained as a closed-set
                three-class model. Its softmax outputs are not calibrated
                probabilities and cannot identify out-of-distribution images.
                Classification is blocked until an independently validated
                ultrasound eligibility detector is available.

                No clinical diagnosis, lesion detection, or lesion sizing is
                performed. Grad-CAM is not a validated lesion localization tool.
                """
            )


def main() -> None:
    render_header()
    render_hero()

    with st.container(key="upload_card"):
        st.markdown(
            """
            <div class="upload-icon">⇧</div>
            <p class="upload-heading">Upload Breast Ultrasound Image</p>
            <p class="upload-copy">Drag and drop your image here, or click to browse</p>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload Breast Ultrasound Image",
            type=["png", "jpg", "jpeg"],
            help="Choose one breast ultrasound image to analyze.",
            label_visibility="collapsed",
        )
        st.caption("Supports PNG, JPG (JPEG) · Max size: 200MB per file")
        image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        analyze_clicked = st.button(
            "↑   Analyze",
            type="primary",
            use_container_width=True,
            disabled=image_bytes is None,
        )
        st.markdown(
            '<div class="upload-note">ⓘ &nbsp; Classification is disabled until a separate breast-ultrasound eligibility detector has been validated.</div>',
            unsafe_allow_html=True,
        )

    if analyze_clicked and image_bytes is not None:
        try:
            uploaded_image = decode_rgb_image(image_bytes)
            require_verified_ultrasound(uploaded_image)
        except UnsupportedImageError as exc:
            st.error(str(exc))

    if image_bytes is None:
        st.markdown(
            '<p class="upload-copy">Upload an image to see the safety gate. No classifications are currently produced.</p>',
            unsafe_allow_html=True,
        )

    render_feature_strip()
    render_advanced_sections()


if __name__ == "__main__":
    main()
