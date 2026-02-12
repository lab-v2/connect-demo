import json
import base64
import mimetypes
import re
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
SURVEY_QUESTIONS_DIR = DATA_DIR / "survey"
SURVEY_SCORES_DIR = DATA_DIR / "survey_scores"
RULES_DIR = DATA_DIR / "rules"
INTERMEDIATE_OUTPUT_DIR = DATA_DIR / "intermediate_output"

SURVEY_QUESTIONS_PATH = SURVEY_QUESTIONS_DIR / "survey_questions.json"
LEGACY_SURVEY_SCORES_PATH = SURVEY_QUESTIONS_DIR / "survey_scores.json"
LEGACY_RULES_PATH = RULES_DIR / "rules_fired.txt"
LEGACY_SNIPPETS_PATH = INTERMEDIATE_OUTPUT_DIR / "snippets.txt"
LEGACY_CHAT_OUTPUT_PATH = OUTPUT_DIR / "chat_output.log"
ADDITIONAL_Q_PATH = INPUT_DIR / "additional_survey_questions.txt"
ADDITIONAL_RULES_PATH = INPUT_DIR / "additional_rules.txt"
LOGO_DIR = INPUT_DIR / "logos"

MODEL_OPTIONS = [
    "gpt-4o",
    "gpt-5.2",
    "xai/grok-4-fast-reasoning",
    "claude-sonnet-4-5",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0",
    "bedrock/us.deepseek.r1-v1:0",
]
SCENARIO_OPTIONS = ["individualistic", "collectivistic"]

TOP_BOX_HEIGHT = 180
LARGE_BOX_HEIGHT = 380
SMALL_BOX_HEIGHT = 140
CHAT_OUTPUT_HEIGHT = 600


def ensure_dirs() -> None:
    for directory in [
        INPUT_DIR,
        OUTPUT_DIR,
        SURVEY_QUESTIONS_DIR,
        SURVEY_SCORES_DIR,
        RULES_DIR,
        INTERMEDIATE_OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODEL_OPTIONS:
        for scenario in SCENARIO_OPTIONS:
            paths = selection_paths(model, scenario)
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_text(path: Path, fallback: str = "") -> str:
    if not path.exists():
        return fallback
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return fallback


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def get_score(scores: dict, key: str) -> str:
    if key in scores:
        return str(scores[key])
    return "N/A"


def get_questions(questions_data: dict) -> list[str]:
    questions = questions_data.get("questions", [])
    if isinstance(questions, list):
        return [str(q) for q in questions]
    return []


def append_questions_to_survey_json(raw_text: str) -> int:
    incoming = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not incoming:
        return 0
    existing_payload = load_json(SURVEY_QUESTIONS_PATH)
    existing_questions = get_questions(existing_payload)
    updated = existing_questions + incoming
    save_json(SURVEY_QUESTIONS_PATH, {"questions": updated})
    return len(incoming)


def append_questions_to_input_txt(raw_text: str) -> int:
    incoming = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not incoming:
        return 0
    current = load_text(ADDITIONAL_Q_PATH, "").strip()
    lines = [line for line in current.splitlines() if line.strip()]
    lines.extend(incoming)
    save_text(ADDITIONAL_Q_PATH, "\n".join(lines) + "\n")
    return len(incoming)


def slugify_model(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def selection_paths(model: str, scenario: str) -> dict[str, Path]:
    model_key = slugify_model(model)
    scenario_key = scenario.strip().lower()
    return {
        "survey_scores": SURVEY_SCORES_DIR / model_key / scenario_key / "survey_scores.json",
        "rules": RULES_DIR / model_key / scenario_key / "rules_fired.txt",
        "snippets": INTERMEDIATE_OUTPUT_DIR / model_key / scenario_key / "snippets.txt",
        "chat_output": OUTPUT_DIR / model_key / scenario_key / "chat_output.log",
    }


def seed_selection_data() -> None:
    legacy_scores = load_json(LEGACY_SURVEY_SCORES_PATH)
    legacy_snippets = load_text(LEGACY_SNIPPETS_PATH, "")
    legacy_chat_output = load_text(LEGACY_CHAT_OUTPUT_PATH, "")
    for model in MODEL_OPTIONS:
        for scenario in SCENARIO_OPTIONS:
            paths = selection_paths(model, scenario)
            if not paths["survey_scores"].exists() and legacy_scores:
                save_json(paths["survey_scores"], legacy_scores)
            if not paths["rules"].exists():
                scenario_legacy = RULES_DIR / f"{scenario}_rules.txt"
                rules_seed = load_text(scenario_legacy, "")
                if not rules_seed:
                    rules_seed = load_text(LEGACY_RULES_PATH, "")
                save_text(paths["rules"], rules_seed)
            if not paths["snippets"].exists():
                save_text(paths["snippets"], legacy_snippets)
            if not paths["chat_output"].exists():
                save_text(paths["chat_output"], legacy_chat_output)


def resolve_logo_src(preferred_name: str, fallback_url: str) -> str:
    local_candidates = [
        LOGO_DIR / preferred_name,
        LOGO_DIR / f"{Path(preferred_name).stem}.jpg",
        LOGO_DIR / f"{Path(preferred_name).stem}.jpeg",
        LOGO_DIR / f"{Path(preferred_name).stem}.svg",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return file_to_data_uri(candidate)
    return fallback_url


def file_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "application/octet-stream"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def svg_to_data_uri(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render() -> None:
    ensure_dirs()
    seed_selection_data()
    st.set_page_config(page_title="Connect Demo", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --su-orange: #f76900;
            --su-navy: #000e54;
            --su-bg: #fffaf5;
            --su-border: #ffd7bd;
        }
        .stApp {
            background: linear-gradient(180deg, #fff3e9 0%, var(--su-bg) 45%, #ffffff 100%);
        }
        h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText, p, span, div {
            color: #1f1f1f;
        }
        .stButton > button {
            background: var(--su-orange);
            color: white;
            border: 1px solid var(--su-orange);
            font-weight: 600;
        }
        .stButton > button:hover {
            background: #de5f00;
            border-color: #de5f00;
            color: white;
        }
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            border: 1px solid var(--su-border);
            border-radius: 8px;
        }
        .stTextArea textarea[disabled] {
            background: #fffdfb;
            color: #1f1f1f;
        }
        .uni-logo-strip {
            position: fixed;
            right: 16px;
            bottom: 16px;
            z-index: 9999;
            display: flex;
            gap: 10px;
            align-items: center;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid var(--su-border);
            border-radius: 12px;
            padding: 8px 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        }
        .uni-logo-strip img {
            height: 34px;
            width: auto;
            object-fit: contain;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Connect Demo")

    if "cleared_inputs_on_start" not in st.session_state:
        save_text(ADDITIONAL_Q_PATH, "")
        save_text(ADDITIONAL_RULES_PATH, "")
        st.session_state.cleared_inputs_on_start = True

    questions_data = load_json(SURVEY_QUESTIONS_PATH)
    saved_additional = load_text(ADDITIONAL_Q_PATH, "")
    saved_additional_rules = load_text(ADDITIONAL_RULES_PATH, "")

    left_col, middle_col, right_col = st.columns([2, 2, 1], gap="medium")

    with left_col:
        chat_input = st.text_area("Input Story", height=TOP_BOX_HEIGHT, key="chat_input")

        c1, c2, c3 = st.columns([1, 1, 1], gap="small")
        with c1:
            model = st.selectbox("Model", MODEL_OPTIONS)
        with c2:
            scenario = st.selectbox("Scenario", SCENARIO_OPTIONS)
        with c3:
            send_clicked = st.button("Transform", use_container_width=True)

        selected_paths = selection_paths(model, scenario)
        scores_data = load_json(selected_paths["survey_scores"])
        rules_text = load_text(selected_paths["rules"], "No rules loaded.")
        snippets_text = load_text(selected_paths["snippets"], "No snippets loaded.")

        if send_clicked and chat_input.strip():
            response = "Placeholder output panel for your backend response."
            block = (
                f"Model: {model} | Scenario: {scenario}\n"
                f"Input: {chat_input.strip()}\n"
                f"Response: {response}\n"
            )
            append_text(selected_paths["chat_output"], block + "\n")
            st.session_state.chat_input = ""
            st.rerun()

        chat_output = load_text(selected_paths["chat_output"], "").strip()
        st.text_area("Transformed Story", value=chat_output, height=CHAT_OUTPUT_HEIGHT, disabled=True)

    with middle_col:
        top_left, top_right = st.columns([2, 1], gap="small")

        with top_left:
            additional_q = st.text_area(
                "Add Survey Questions",
                value=saved_additional,
                height=TOP_BOX_HEIGHT,
                key="additional_questions",
            )
            if st.button("Extend Survey", use_container_width=True):
                added_count = append_questions_to_survey_json(additional_q)
                append_questions_to_input_txt(additional_q)
                if added_count > 0:
                    st.success(
                        f"Appended {added_count} question(s) to survey_questions.json and additional_survey_questions.txt"
                    )
                    st.rerun()
                else:
                    st.warning("No non-empty questions to append.")

        with top_right:
            st.text_area(
                "Original Story Score",
                value=get_score(scores_data, "original_story"),
                height=TOP_BOX_HEIGHT,
                disabled=True,
            )

        questions = get_questions(questions_data)
        if questions:
            questions_text = "\n\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        else:
            questions_text = "No survey questions found in JSON."
        st.text_area(
            "Survey Questions (From JSON)",
            value=questions_text,
            height=LARGE_BOX_HEIGHT,
            disabled=True,
        )
        st.text_area(
            "Transformed Story Score",
            value=get_score(scores_data, "transformed_story"),
            height=SMALL_BOX_HEIGHT,
            disabled=True,
        )

    with right_col:
        additional_rules = st.text_area(
            "Add Rules",
            value=saved_additional_rules,
            height=TOP_BOX_HEIGHT,
            key="additional_rules",
        )
        if st.button("Add Rules", use_container_width=True):
            save_text(ADDITIONAL_RULES_PATH, additional_rules.strip())
            st.success(f"Saved to {ADDITIONAL_RULES_PATH.relative_to(BASE_DIR)}")
        st.text_area("Rules Fired (TXT)", value=rules_text, height=LARGE_BOX_HEIGHT, disabled=True)
        st.text_area("Text Snippets (TXT)", value=snippets_text, height=SMALL_BOX_HEIGHT, disabled=True)

    syracuse_fallback_svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='220' height='70' viewBox='0 0 220 70'>
      <rect width='220' height='70' rx='10' fill='#F76900'/>
      <text x='110' y='30' text-anchor='middle' font-size='16' font-family='Arial, sans-serif' fill='white' font-weight='700'>SYRACUSE</text>
      <text x='110' y='50' text-anchor='middle' font-size='12' font-family='Arial, sans-serif' fill='white'>UNIVERSITY</text>
    </svg>
    """
    asu_fallback_svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='220' height='70' viewBox='0 0 220 70'>
      <rect width='220' height='70' rx='10' fill='#8C1D40'/>
      <text x='110' y='30' text-anchor='middle' font-size='20' font-family='Arial, sans-serif' fill='#FFC627' font-weight='700'>ASU</text>
      <text x='110' y='50' text-anchor='middle' font-size='12' font-family='Arial, sans-serif' fill='white'>ARIZONA STATE UNIVERSITY</text>
    </svg>
    """
    syracuse_logo = resolve_logo_src("syracuse_logo.png", svg_to_data_uri(syracuse_fallback_svg))
    asu_logo = resolve_logo_src("asu_logo.png", svg_to_data_uri(asu_fallback_svg))
    st.markdown(
        f"""
        <div class="uni-logo-strip">
            <img src="{syracuse_logo}" alt="Syracuse University logo" />
            <img src="{asu_logo}" alt="Arizona State University logo" />
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
