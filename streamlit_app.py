import json
import base64
import mimetypes
import hashlib
import re
import subprocess
import os
import statistics
import math
import html
import shutil
from difflib import SequenceMatcher
from pathlib import Path
from collections import Counter

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
SURVEY_QUESTIONS_DIR = DATA_DIR / "survey"
RULES_DIR = DATA_DIR / "rules"

SURVEY_QUESTIONS_PATH = SURVEY_QUESTIONS_DIR / "survey_questions.json"
LEGACY_RULES_PATH = RULES_DIR / "rules_fired.txt"
INPUT_STORY_PATH = INPUT_DIR / "input.txt"
LOGO_DIR = INPUT_DIR / "logos"

DEFAULT_RATING_INSTRUCTIONS = [
    "You are an expert literary analyst.",
    "Given a story, analyze the question using a 1–5 scale to assess the narrative perspective.",
    "Scale:",
    "1 = Entirely individual perspective",
    "2 = Primarily individual but with some group influence",
    "3 = Balanced between individual and group",
    "4 = Primarily group-oriented",
    "5 = Entirely group/community perspective",
    "Identify the TOP sections of the story that demonstrate this quality. Provide the ENTIRE section in quote format, minimum full sentence.",
]

MODEL_OPTIONS = [
    "gpt-4o",
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm2:D5JJuHZi",
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm3:D75Ahi1l",
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm4:D5NuhUdq",
    "gpt-5.2",
    "xai/grok-4-fast-reasoning",
    "claude-sonnet-4-5",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0",
    "bedrock/us.deepseek.r1-v1:0",
]
SCENARIO_OPTIONS = ["individualistic", "collectivistic"]

MODEL_PROVIDER_BY_NAME = {
    "gpt-4o": "openai",
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm2:D5JJuHZi": "openai",
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm3:D75Ahi1l": "openai",
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm4:D5NuhUdq": "openai",
    "gpt-5.2": "openai",
    "xai/grok-4-fast-reasoning": "xai",
    "claude-sonnet-4-5": "anthropic",
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": "bedrock",
    "bedrock/us.deepseek.r1-v1:0": "bedrock",
}

PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "bedrock": None,
}

TOP_BOX_HEIGHT = 200
LARGE_BOX_HEIGHT = 520
SMALL_BOX_HEIGHT = 220
CHAT_OUTPUT_HEIGHT = 700
SURVEY_BOX_HEIGHT = 180
ADDITIONAL_SURVEY_BOX_HEIGHT = 110
PHASE2_ANALYSIS_ITERATIONS = 2
PHASE2_TOTAL_ITERATIONS = PHASE2_ANALYSIS_ITERATIONS + 1


def model_provider(model: str) -> str:
    explicit = MODEL_PROVIDER_BY_NAME.get(model)
    if explicit:
        return explicit
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("xai/"):
        return "xai"
    if model.startswith("anthropic/") or model.startswith("claude"):
        return "anthropic"
    if model.startswith("huggingface/") or model.startswith("hf/"):
        return "huggingface"
    if model.startswith("bedrock/") or model.startswith("qwen"):
        return "bedrock"
    return "openai"


def required_api_env_var(model: str) -> str | None:
    return PROVIDER_ENV_VAR[model_provider(model)]


def apply_startup_credentials(model: str, provider_api_key: str, hf_token: str) -> None:
    env_var = required_api_env_var(model)
    if env_var and provider_api_key:
        os.environ[env_var] = provider_api_key
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token


def require_startup_config() -> tuple[str, str]:
    setup_complete = st.session_state.get("_setup_complete", False)
    if setup_complete:
        selected_model = st.session_state.get("_setup_model", MODEL_OPTIONS[0])
        provider_api_key = st.session_state.get("_setup_provider_api_key", "")
        hf_token = st.session_state.get("_setup_hf_token", "")
        apply_startup_credentials(selected_model, provider_api_key, hf_token)
        provider = model_provider(selected_model)
        return selected_model, provider

    st.title("NARRATE Setup")
    st.markdown("Select a model and provide required credentials to continue.")

    selected_model = st.selectbox("Model", MODEL_OPTIONS, key="_setup_model_draft")
    provider = model_provider(selected_model)
    required_var = required_api_env_var(selected_model)
    st.caption(f"Provider for selected model: `{provider}`")
    provider_api_key = st.text_input(
        "Provider API Key",
        type="password",
        key="_setup_provider_key_draft",
    )
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        key="_setup_hf_token_draft",
    )
    submitted = st.button("Start", use_container_width=True)

    if submitted:
        provider_api_key = (provider_api_key or "").strip()
        hf_token = (hf_token or "").strip()

        if required_var and required_var != "HF_TOKEN" and not provider_api_key:
            st.error("Please provide a provider API key.")
            st.stop()
        if required_var == "HF_TOKEN" and not hf_token and not provider_api_key:
            st.error("Please provide a Hugging Face token.")
            st.stop()

        st.session_state["_setup_complete"] = True
        st.session_state["_setup_model"] = selected_model
        st.session_state["_setup_provider_api_key"] = provider_api_key if required_var != "HF_TOKEN" else (provider_api_key or hf_token)
        st.session_state["_setup_hf_token"] = hf_token or (provider_api_key if required_var == "HF_TOKEN" else "")
        apply_startup_credentials(
            st.session_state["_setup_model"],
            st.session_state["_setup_provider_api_key"],
            st.session_state["_setup_hf_token"],
        )
        st.rerun()

    st.stop()


def ensure_dirs() -> None:
    for directory in [
        INPUT_DIR,
        SURVEY_QUESTIONS_DIR,
        RULES_DIR,
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


def format_score_one_decimal(value: str | int | float | None) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return "N/A"
    try:
        return f"{float(text):.1f}"
    except Exception:
        return text


def format_score_integer(value: str | int | float | None) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return "N/A"
    try:
        return str(int(round(float(text))))
    except Exception:
        return text


def get_questions(questions_data: dict) -> list[str]:
    questions = questions_data.get("questions", [])
    if isinstance(questions, list):
        return [str(q) for q in questions]
    return []


def append_questions_to_survey_json(raw_text: str, target_path: Path) -> int:
    incoming = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not incoming:
        return 0
    existing_payload = load_json(target_path)
    existing_questions = get_questions(existing_payload)
    updated = existing_questions + incoming
    save_json(target_path, {"questions": updated})
    return len(incoming)


def additional_survey_path_for_scenario(scenario: str) -> Path:
    scenario_key = scenario.strip().lower()
    return SURVEY_QUESTIONS_DIR / scenario_key / "additional_survey.json"


def parse_additional_questions_blocks(raw_text: str) -> list[tuple[str, str]]:
    blocks = re.split(r"\n\s*\n+", raw_text.strip())
    parsed: list[tuple[str, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        component = lines[0]
        question = " ".join(lines[1:]).strip()
        if component and question:
            parsed.append((component, question))
    return parsed


def append_additional_survey_json(raw_text: str, scenario: str, target_path: Path) -> int:
    parsed_blocks = parse_additional_questions_blocks(raw_text)
    if not parsed_blocks:
        return 0

    question_key = "collectivistic_question" if scenario == "collectivistic" else "individualistic_question"

    payload = load_json(target_path)
    existing = payload.get("questions", [])
    if not isinstance(existing, list):
        existing = []

    max_id = 0
    base_path = survey_questions_path_for_scenario(scenario)
    base_payload = load_json(base_path)
    base_questions = base_payload.get("questions", []) if isinstance(base_payload, dict) else []
    if isinstance(base_questions, list):
        for item in base_questions:
            if isinstance(item, dict):
                try:
                    max_id = max(max_id, int(item.get("id", 0)))
                except Exception:
                    pass
    for item in existing:
        if isinstance(item, dict):
            try:
                max_id = max(max_id, int(item.get("id", 0)))
            except Exception:
                pass

    next_id = max_id + 2 if max_id else 2
    new_items = []
    for component, question in parsed_blocks:
        new_items.append(
            {
                "id": next_id,
                "component": component,
                question_key: {
                    "question": question,
                    "type": "ordinal_scale",
                    "rating": DEFAULT_RATING_INSTRUCTIONS,
                },
            }
        )
        next_id += 2

    updated = {"questions": existing + new_items}
    save_json(target_path, updated)
    return len(new_items)


def ensure_additional_survey_files_exist() -> None:
    """Ensure additional-survey files exist without clearing existing content."""
    for scenario_name in SCENARIO_OPTIONS:
        p = additional_survey_path_for_scenario(scenario_name)
        if not p.exists():
            save_json(p, {"questions": []})


def slugify_model(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def sanitize_model_for_phase2(model: str) -> str:
    # Match src/llm_survey.py sanitize_model_name behavior used by src/main.py
    return model.replace("/", "-").replace(":", "-")


def sanitize_story_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "input"


def survey_questions_path_for_scenario(scenario: str) -> Path:
    scenario_key = scenario.strip().lower()
    return SURVEY_QUESTIONS_DIR / scenario_key / "survey_questions.json"


def selection_paths(model: str, scenario: str) -> dict[str, Path]:
    model_key = slugify_model(model)
    scenario_key = scenario.strip().lower()
    return {
        "rules": RULES_DIR / model_key / scenario_key / "rules_fired.txt",
        "pyreason_rules": RULES_DIR / model_key / scenario_key / "pyreason_rules.txt",
        "selected_rules": RULES_DIR / model_key / scenario_key / "selected_rules.txt",
    }


def phase2_problem_for_scenario(scenario: str) -> str:
    return "forward" if scenario == "individualistic" else "inverse"


def llm_survey_questions_file_for_scenario(scenario: str) -> Path:
    scenario_key = scenario.strip().lower()
    return DATA_DIR / f"{scenario_key}_questions.json"


def display_questions_for_scenario(scenario: str) -> list[str]:
    questions_file = llm_survey_questions_file_for_scenario(scenario)
    problem_type = phase2_problem_for_scenario(scenario)
    question_key = "individualistic_question" if problem_type == "forward" else "collectivistic_question"

    try:
        from src.llm_survey import load_questions

        merged_questions = load_questions(
            str(questions_file),
            include_additional_questions=True,
        )
        out: list[str] = []
        for item in merged_questions:
            if not isinstance(item, dict):
                continue
            qobj = item.get(question_key)
            if isinstance(qobj, dict):
                qtext = qobj.get("question")
                if isinstance(qtext, str) and qtext.strip():
                    out.append(qtext.strip())
        return out
    except Exception:
        pass

    fallback_payload = load_json(survey_questions_path_for_scenario(scenario))
    return get_questions(fallback_payload)


def extract_ratings(payload: dict) -> list[float]:
    ratings: list[float] = []
    for item in payload.get("questions_and_answers", []):
        if not isinstance(item, dict):
            continue
        value = item.get("rating")
        if isinstance(value, (int, float)):
            ratings.append(float(value))
        elif isinstance(value, str):
            try:
                ratings.append(float(value.strip()))
            except Exception:
                pass
    return ratings


def median_from_survey_json(path: Path) -> str:
    payload = load_json(path)
    ratings = extract_ratings(payload)
    if not ratings:
        return "N/A"
    return f"{statistics.median(ratings):.2f}"


def median_from_survey_result(payload: dict) -> str:
    ratings = extract_ratings(payload)
    if not ratings:
        return "N/A"
    return f"{statistics.median(ratings):.2f}"


def mean_from_survey_json(path: Path | None) -> float | None:
    if not path:
        return None
    payload = load_json(path)
    ratings = extract_ratings(payload)
    if not ratings:
        return None
    return sum(ratings) / len(ratings)


def find_phase2_iteration_survey(
    model: str,
    scenario: str,
    iteration: int,
    story_name: str | None = None,
    min_mtime: float | None = None,
) -> Path | None:
    problem = phase2_problem_for_scenario(scenario)
    model_dir = sanitize_model_for_phase2(model)
    base = BASE_DIR / "output" / "phase2" / model_dir / problem

    def is_fresh(path: Path) -> bool:
        if not path.exists():
            return False
        if min_mtime is None:
            return True
        return path.stat().st_mtime >= min_mtime

    if story_name:
        preferred = base / story_name / f"iteration_{iteration}" / "survey.json"
        if is_fresh(preferred):
            return preferred
        candidates = [p for p in sorted(base.glob(f"{story_name}/iteration_{iteration}/survey.json")) if is_fresh(p)]
    else:
        direct = base / f"iteration_{iteration}" / "survey.json"
        if is_fresh(direct):
            return direct
        candidates = [p for p in sorted(base.glob(f"*/iteration_{iteration}/survey.json")) if is_fresh(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_phase2_iteration_file(
    model: str,
    scenario: str,
    iteration: int,
    filename: str,
    story_name: str | None = None,
    min_mtime: float | None = None,
) -> Path | None:
    problem = phase2_problem_for_scenario(scenario)
    model_dir = sanitize_model_for_phase2(model)
    base = BASE_DIR / "output" / "phase2" / model_dir / problem

    def is_fresh(path: Path) -> bool:
        if not path.exists():
            return False
        if min_mtime is None:
            return True
        return path.stat().st_mtime >= min_mtime

    if story_name:
        preferred = base / story_name / f"iteration_{iteration}" / filename
        if is_fresh(preferred):
            return preferred
        candidates = [p for p in sorted(base.glob(f"{story_name}/iteration_{iteration}/{filename}")) if is_fresh(p)]
    else:
        direct = base / f"iteration_{iteration}" / filename
        if is_fresh(direct):
            return direct
        candidates = [p for p in sorted(base.glob(f"*/iteration_{iteration}/{filename}")) if is_fresh(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_optimal_iteration_for_dashboard(
    model: str,
    scenario: str,
    max_scan: int = 20,
    story_name: str | None = None,
    min_mtime: float | None = None,
) -> int:
    """
    Choose optimal iteration using the same objective as find_optimal_iteration.py:
      - forward: minimum mean rating
      - inverse: maximum mean rating
    """
    scored: dict[int, float] = {}
    for i in range(max_scan):
        survey_path = find_phase2_iteration_survey(
            model,
            scenario,
            i,
            story_name=story_name,
            min_mtime=min_mtime,
        )
        mean_val = mean_from_survey_json(survey_path)
        if mean_val is not None:
            scored[i] = mean_val

    if not scored:
        return 2

    problem = phase2_problem_for_scenario(scenario)
    if problem == "forward":
        return min(scored, key=scored.get)
    return max(scored, key=scored.get)


def transformed_story_text(
    model: str,
    scenario: str,
    iteration: int = 2,
    story_name: str | None = None,
    min_mtime: float | None = None,
) -> str:
    story_path = find_phase2_iteration_file(
        model,
        scenario,
        iteration,
        "story_transformed.txt",
        story_name=story_name,
        min_mtime=min_mtime,
    )
    if not story_path:
        return "No transformed story found."
    text = load_text(story_path, "").strip()
    return text if text else "No transformed story found."


def recompute_scores_with_extended_survey(
    model: str,
    scenario: str,
    original_text: str,
    transformed_text: str,
) -> tuple[str, str, float]:
    from src.llm_survey import CostTracker, conduct_survey_single_story

    questions_file = llm_survey_questions_file_for_scenario(scenario)
    problem_type = phase2_problem_for_scenario(scenario)
    cost_tracker = CostTracker(model)

    original_story = {
        "name": "input",
        "path": str(INPUT_STORY_PATH),
        "content": original_text,
    }
    transformed_story = {
        "name": "story_transformed",
        "path": "story_transformed.txt",
        "content": transformed_text,
    }

    original_result = conduct_survey_single_story(
        story=original_story,
        questions_file=str(questions_file),
        problem_type=problem_type,
        model=model,
        temperature=0.7,
        cost_tracker=cost_tracker,
        include_additional_questions=True,
    )
    transformed_result = conduct_survey_single_story(
        story=transformed_story,
        questions_file=str(questions_file),
        problem_type=problem_type,
        model=model,
        temperature=0.7,
        cost_tracker=cost_tracker,
        include_additional_questions=True,
    )

    return (
        median_from_survey_result(original_result),
        median_from_survey_result(transformed_result),
        cost_tracker.total_cost,
    )


def tokenize_for_divergence(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


def token_distribution(tokens: list[str], smoothing: float = 1e-5) -> dict[str, float]:
    if not tokens:
        return {}
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    vocab_size = len(token_counts)
    return {
        token: (count + smoothing) / (total_tokens + smoothing * vocab_size)
        for token, count in token_counts.items()
    }


def kl_transformed_vs_original(transformed_text: str, original_text: str, smoothing: float = 1e-5) -> float | None:
    transformed_dist = token_distribution(tokenize_for_divergence(transformed_text), smoothing)
    original_dist = token_distribution(tokenize_for_divergence(original_text), smoothing)
    if not transformed_dist or not original_dist:
        return None

    all_tokens = set(transformed_dist.keys()) | set(original_dist.keys())
    kl_sum = 0.0
    for token in all_tokens:
        p = transformed_dist.get(token, smoothing)
        q = original_dist.get(token, smoothing)
        kl_sum += p * math.log(p / q)
    return float(kl_sum)


def bertscore_ab_f1(transformed_text: str, original_text: str) -> float | None:
    if not transformed_text.strip() or not original_text.strip():
        return None
    try:
        from src.bertscore_analysis import compute_bertscore

        _precision, _recall, f1 = compute_bertscore(original_text, transformed_text)
        if f1 is None:
            return None
        return float(f1)
    except Exception:
        return None


def phase2_iteration_dir(
    model: str,
    scenario: str,
    iteration: int,
    story_name: str | None = None,
) -> Path:
    problem = phase2_problem_for_scenario(scenario)
    model_dir = sanitize_model_for_phase2(model)
    base = BASE_DIR / "output" / "phase2" / model_dir / problem
    if story_name:
        return base / story_name / f"iteration_{iteration}"
    return base / f"iteration_{iteration}"


def semantic_similarity_from_cache_or_compute(
    model: str,
    scenario: str,
    story_name: str | None,
    iteration: int,
    original_text: str,
    transformed_text: str,
) -> float | None:
    problem = phase2_problem_for_scenario(scenario)
    original_clean = (original_text or "").strip()
    transformed_clean = (transformed_text or "").strip()
    if not original_clean or not transformed_clean or transformed_clean == "No transformed story found.":
        return None

    cache_path = phase2_iteration_dir(
        model,
        scenario,
        iteration,
        story_name=story_name,
    ) / "semantic_similarity.json"
    original_sha1 = hashlib.sha1(original_clean.encode("utf-8")).hexdigest()
    transformed_sha1 = hashlib.sha1(transformed_clean.encode("utf-8")).hexdigest()

    cached = load_json(cache_path)
    if isinstance(cached, dict):
        if (
            cached.get("metric") == "bertscore_f1"
            and cached.get("original_text_sha1") == original_sha1
            and cached.get("transformed_text_sha1") == transformed_sha1
        ):
            try:
                return float(cached.get("semantic_similarity_f1"))
            except Exception:
                pass

    semantic_similarity_value = bertscore_ab_f1(transformed_clean, original_clean)
    payload = {
        "metric": "bertscore_f1",
        "semantic_similarity_f1": semantic_similarity_value,
        "original_text_sha1": original_sha1,
        "transformed_text_sha1": transformed_sha1,
        "model": model,
        "problem": problem,
        "story_name": story_name or "",
        "iteration": iteration,
    }
    save_json(cache_path, payload)
    return semantic_similarity_value


def avg_current_confidence_from_abduction(path: Path | None) -> float | None:
    if not path:
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    feature_gaps = payload.get("feature_gaps", [])
    if not isinstance(feature_gaps, list):
        return None

    confidences: list[float] = []
    for gap in feature_gaps:
        if not isinstance(gap, dict):
            continue
        current_confidence = gap.get("current_confidence")
        try:
            confidences.append(float(current_confidence))
        except Exception:
            continue

    if not confidences:
        return None
    return sum(confidences) / len(confidences)


def explanations_from_ranked_prescriptions(path: Path | None) -> str:
    if not path:
        return ""
    payload = load_json(path)
    prescriptions = payload.get("prescriptions", []) if isinstance(payload, dict) else []
    if not isinstance(prescriptions, list) or not prescriptions:
        return ""

    lines: list[str] = []
    for item in prescriptions:
        if not isinstance(item, dict):
            continue
        segment_text = str(item.get("segment_text", "")).strip()
        feature = str(item.get("feature", "")).strip()
        if not segment_text or not feature:
            continue
        lines.append(
            f"We abduce the segment {segment_text} based on the narrative characteristic of {feature}."
        )
    return "\n\n".join(lines)


def segments_from_ranked_prescriptions(path: Path | None) -> list[str]:
    if not path:
        return []
    payload = load_json(path)
    prescriptions = payload.get("prescriptions", []) if isinstance(payload, dict) else []
    if not isinstance(prescriptions, list):
        return []
    segments: list[str] = []
    seen = set()
    for item in prescriptions:
        if not isinstance(item, dict):
            continue
        segment = str(item.get("segment_text", "")).strip()
        if not segment:
            continue
        if segment not in seen:
            seen.add(segment)
            segments.append(segment)
    return segments


def highlighted_story_html(story_text: str, segments: list[str], box_class: str = "original-box") -> str:
    if not story_text.strip():
        return f"<div class='story-box {box_class}'>No story available.</div>"
    if not segments:
        return f"<div class='story-box {box_class}'>{html.escape(story_text)}</div>"

    cleaned = [s.strip().strip("\"'") for s in segments if s.strip().strip("\"'")]
    cleaned = sorted(set(cleaned), key=len, reverse=True)
    if not cleaned:
        return f"<div class='story-box {box_class}'>{html.escape(story_text)}</div>"

    spans: list[tuple[int, int]] = []

    # Pass 1: direct segment matches (case-insensitive).
    pattern = re.compile("|".join(re.escape(s) for s in cleaned), re.IGNORECASE)
    for match in pattern.finditer(story_text):
        spans.append(match.span())

    # Pass 2: fallback to sentence-level fuzzy overlap if no exact hit.
    if not spans:
        def tokenize(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower()))

        segment_tokens = [tokenize(s) for s in cleaned if s]
        sentence_matches = list(re.finditer(r"[^.!?\n]+[.!?]?", story_text))
        for sent in sentence_matches:
            sent_text = sent.group(0).strip()
            if not sent_text:
                continue
            sent_tokens = tokenize(sent_text)
            if len(sent_tokens) < 4:
                continue
            best_overlap = 0.0
            for seg_tokens in segment_tokens:
                if not seg_tokens:
                    continue
                overlap = len(sent_tokens & seg_tokens) / max(len(sent_tokens), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
            if best_overlap >= 0.35:
                spans.append(sent.span())

    if not spans:
        return f"<div class='story-box {box_class}'>{html.escape(story_text)}</div>"

    # Merge overlapping spans.
    spans.sort()
    merged: list[tuple[int, int]] = []
    start, end = spans[0]
    for s, e in spans[1:]:
        if s <= end:
            end = max(end, e)
        else:
            merged.append((start, end))
            start, end = s, e
    merged.append((start, end))

    chunks: list[str] = []
    last = 0
    for start, end in merged:
        chunks.append(html.escape(story_text[last:start]))
        chunks.append(f"<mark>{html.escape(story_text[start:end])}</mark>")
        last = end
    chunks.append(html.escape(story_text[last:]))
    return f"<div class='story-box {box_class}'>{''.join(chunks)}</div>"


def highlighted_transformed_diff_html(original_text: str, transformed_text: str) -> str:
    if not transformed_text.strip():
        return "<div class='story-box'>No transformed story found.</div>"

    matcher = SequenceMatcher(None, original_text, transformed_text)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "insert"} and j2 > j1:
            spans.append((j1, j2))

    if not spans:
        return f"<div class='story-box'>{html.escape(transformed_text)}</div>"

    spans.sort()
    merged: list[tuple[int, int]] = []
    start, end = spans[0]
    for s, e in spans[1:]:
        if s <= end:
            end = max(end, e)
        else:
            merged.append((start, end))
            start, end = s, e
    merged.append((start, end))

    chunks: list[str] = []
    last = 0
    for start, end in merged:
        chunks.append(html.escape(transformed_text[last:start]))
        chunks.append(f"<mark class='diff'>{html.escape(transformed_text[start:end])}</mark>")
        last = end
    chunks.append(html.escape(transformed_text[last:]))
    return f"<div class='story-box'>{''.join(chunks)}</div>"


def _segment_spans_for_text(text: str, segments: list[str]) -> list[tuple[int, int]]:
    if not text.strip() or not segments:
        return []
    cleaned = [s.strip().strip("\"'") for s in segments if s.strip().strip("\"'")]
    cleaned = sorted(set(cleaned), key=len, reverse=True)
    if not cleaned:
        return []

    spans: list[tuple[int, int]] = []
    pattern = re.compile("|".join(re.escape(s) for s in cleaned), re.IGNORECASE)
    for match in pattern.finditer(text):
        spans.append(match.span())
    return spans


def _diff_spans(original_text: str, transformed_text: str) -> list[tuple[int, int]]:
    matcher = SequenceMatcher(None, original_text, transformed_text)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "insert"} and j2 > j1:
            spans.append((j1, j2))
    return spans


def highlighted_transformed_combined_html(original_text: str, transformed_text: str, ranked_segments: list[str]) -> str:
    if not transformed_text.strip():
        return "<div class='story-box transformed-box'>No transformed story found.</div>"

    ranked_spans = _segment_spans_for_text(transformed_text, ranked_segments)
    diff_spans = _diff_spans(original_text, transformed_text)
    if not ranked_spans and not diff_spans:
        return f"<div class='story-box transformed-box'>{html.escape(transformed_text)}</div>"

    boundaries = {0, len(transformed_text)}
    for s, e in ranked_spans + diff_spans:
        boundaries.add(s)
        boundaries.add(e)
    points = sorted(boundaries)

    def covered(i: int, spans: list[tuple[int, int]]) -> bool:
        for s, e in spans:
            if s <= i < e:
                return True
        return False

    chunks: list[str] = []
    for a, b in zip(points, points[1:]):
        piece = transformed_text[a:b]
        if not piece:
            continue
        in_ranked = covered(a, ranked_spans)
        in_diff = covered(a, diff_spans)
        if in_ranked and in_diff:
            chunks.append(f"<mark class='both'>{html.escape(piece)}</mark>")
        elif in_diff:
            chunks.append(f"<mark class='diff'>{html.escape(piece)}</mark>")
        elif in_ranked:
            chunks.append(f"<mark>{html.escape(piece)}</mark>")
        else:
            chunks.append(html.escape(piece))
    return f"<div class='story-box transformed-box'>{''.join(chunks)}</div>"


def transformed_with_original_replacements_html(original_text: str, transformed_text: str) -> str:
    if not transformed_text.strip():
        return "<div class='story-box transformed-box'>No transformed story found.</div>"

    matcher = SequenceMatcher(None, original_text, transformed_text)
    chunks: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            chunks.append(html.escape(transformed_text[j1:j2]))
            continue

        if tag == "replace":
            original_piece = original_text[i1:i2]
            if original_piece:
                chunks.append(f"<span class='replacement-original'>{html.escape(original_piece)}</span>")
            continue

        if tag == "insert":
            # Hide inserted transformed text in "Hide" mode.
            continue

        if tag == "delete":
            # Bring back deleted original text in "Hide" mode.
            original_piece = original_text[i1:i2]
            if original_piece:
                chunks.append(f"<span class='replacement-original'>{html.escape(original_piece)}</span>")

    return f"<div class='story-box transformed-box'>{''.join(chunks)}</div>"


def seed_selection_data() -> None:
    for model in MODEL_OPTIONS:
        for scenario in SCENARIO_OPTIONS:
            paths = selection_paths(model, scenario)
            if not paths["rules"].exists():
                scenario_legacy = RULES_DIR / f"{scenario}_rules.txt"
                rules_seed = load_text(scenario_legacy, "")
                if not rules_seed:
                    rules_seed = load_text(LEGACY_RULES_PATH, "")
                save_text(paths["rules"], rules_seed)


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


def run_phase2_transform(
    model: str,
    scenario: str,
    rules_path: Path | None = None,
    story_path: Path | None = None,
) -> tuple[bool, str]:
    problem = phase2_problem_for_scenario(scenario)
    effective_story_path = story_path or INPUT_STORY_PATH
    if rules_path is None:
        scenario_dir = "individualistic" if problem == "forward" else "collectivistic"
        rules_path = RULES_DIR / slugify_model(model) / scenario_dir / "pyreason_rules.txt"
    if not rules_path.exists():
        return False, f"Rules file not found: {rules_path}"
    cmd = [
        "python3",
        str(BASE_DIR / "src" / "main.py"),
        "--phase",
        "2",
        "--problem",
        problem,
        "--model",
        model,
        "--data-dir",
        "data",
        "--output-dir",
        "output",
        "--rules",
        str(rules_path),
        "--story",
        str(effective_story_path),
        "--max-iterations",
        str(PHASE2_TOTAL_ITERATIONS),
        "--top-k",
        "5",
    ]
    proc = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode == 0:
        return True, stdout or "Transform completed."
    parts = []
    if stdout:
        parts.append(f"STDOUT:\n{stdout}")
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    detail = "\n\n".join(parts) if parts else "No output."
    return False, f"Transform failed (exit {proc.returncode}).\n{detail}"


def clear_phase2_output_for_selection(
    model: str,
    scenario: str,
    story_name: str | None = None,
) -> None:
    problem = phase2_problem_for_scenario(scenario)
    model_dir = sanitize_model_for_phase2(model)
    target = BASE_DIR / "output" / "phase2" / model_dir / problem
    if story_name:
        target = target / sanitize_story_name(story_name)
    if target.exists():
        shutil.rmtree(target)


def clear_rerun_inputs_for_selection(model: str, scenario: str) -> None:
    save_text(INPUT_STORY_PATH, "")
    problem = phase2_problem_for_scenario(scenario)
    model_dir = sanitize_model_for_phase2(model)
    input_path = BASE_DIR / "output" / "phase2" / model_dir / problem / "input"
    if input_path.is_dir():
        shutil.rmtree(input_path)
    elif input_path.exists():
        input_path.unlink()


def clear_outputs_on_selection_change(model: str, scenario: str) -> None:
    """
    Clear only when the active model/scenario selection changes.
    Prevents wiping fresh outputs on normal Streamlit reruns.
    """
    prev_model = st.session_state.get("_active_model")
    prev_scenario = st.session_state.get("_active_scenario")

    if prev_model is None or prev_scenario is None:
        st.session_state["_active_model"] = model
        st.session_state["_active_scenario"] = scenario
        return

    if prev_model != model or prev_scenario != scenario:
        clear_rerun_inputs_for_selection(model, scenario)
        st.session_state.hide_transformed_text = False
        st.session_state.pop("extended_survey_scores", None)

    st.session_state["_active_model"] = model
    st.session_state["_active_scenario"] = scenario


def render() -> None:
    ensure_dirs()
    seed_selection_data()
    st.set_page_config(
        page_title="NARRATE: Neurosymbolic Abductive Reasoning for Reframing Texts",
        page_icon="🟠",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --su-orange: #f76900;
            --su-navy: #000e54;
            --su-bg: #fffaf5;
            --su-border: #ffd7bd;
            --box-bg: #fffdfb;
            --box-shadow: 0 6px 16px rgba(0, 14, 84, 0.08);
        }
        .stApp {
            background: linear-gradient(180deg, #fff3e9 0%, var(--su-bg) 45%, #ffffff 100%);
        }
        div[data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            height: 100%;
        }
        h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText, p, span, div {
            color: #1f1f1f;
        }
        .stButton > button {
            background: #f7c8a8;
            color: #5a3522;
            border: 1px solid #e9b693;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: #efb894;
            border-color: #ddab88;
            color: #4d2f1f;
        }
        .stButton > button[aria-label="Reframe"] {
            clip-path: polygon(0 0, 88% 0, 100% 50%, 88% 100%, 0 100%, 10% 50%);
            padding-left: 10px;
            padding-right: 10px;
            font-weight: 900 !important;
            display: block;
            margin: 0 auto;
            width: 100%;
            min-height: 78px;
            font-size: 1.38rem;
            letter-spacing: 0.01em;
            white-space: nowrap;
            line-height: 1;
        }
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
            border: 1px solid var(--su-border);
            border-radius: 12px;
            background: var(--box-bg);
            box-shadow: var(--box-shadow);
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] > div {
            visibility: hidden;
            position: relative;
            min-height: 24px;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] > div::after {
            content: "Upload Input Story";
            visibility: visible;
            position: absolute;
            left: 0;
            right: 0;
            text-align: center;
            font-weight: 600;
            color: #1f1f1f;
        }
        div[data-testid="stFileUploader"] {
            margin-top: 0.9rem;
            margin-bottom: 0.6rem;
        }
        div[data-testid="stFileUploaderDropzone"] {
            border: 2px dashed #e8b995;
            border-radius: 0;
            background: linear-gradient(180deg, #fffaf6 0%, #fff4ea 100%);
            min-height: 124px;
            padding-top: 0.85rem;
            padding-bottom: 0.85rem;
            box-shadow: 0 8px 20px rgba(247, 105, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.8);
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        }
        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #d99d74;
            box-shadow: 0 10px 24px rgba(247, 105, 0, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.9);
            transform: translateY(-1px);
        }
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            overflow-x: auto;
            overflow-y: auto;
            white-space: nowrap;
            min-height: 56px;
            max-height: 84px;
            font-size: 15px;
            border-radius: 12px !important;
            background: var(--box-bg);
            box-shadow: var(--box-shadow);
        }
        div[data-baseweb="select"] span[data-baseweb="tag"] {
            display: block;
            width: 100%;
            border-radius: 6px;
            padding: 2px 6px;
            margin: 0;
            min-height: 0;
            white-space: nowrap;
            overflow-x: auto;
            overflow-y: hidden;
            line-height: 1.1;
            font-size: 14px;
            background: #f8e6da !important;
            border: 1px solid #e7c6b2 !important;
            color: #5f3d2b !important;
        }
        div[role="listbox"] {
            max-height: 560px !important;
        }
        div[role="option"] {
            min-height: 52px !important;
            white-space: normal !important;
            overflow-y: hidden !important;
            line-height: 1.4 !important;
            font-size: 14px !important;
            padding-top: 12px !important;
            padding-bottom: 12px !important;
        }
        .stTextArea textarea[disabled] {
            background: var(--box-bg);
            color: #1f1f1f;
        }
        .story-box {
            border: 1px solid var(--su-border);
            border-radius: 12px;
            background: var(--box-bg);
            padding: 12px;
            height: 260px;
            white-space: pre-wrap;
            overflow-y: auto;
            line-height: 1.35;
            box-shadow: var(--box-shadow);
        }
        .story-box.original-box {
            height: 170px;
            width: 100%;
        }
        .story-box.transformed-box {
            height: 560px !important;
            width: 100% !important;
            max-width: none;
            margin: 0;
            box-sizing: border-box;
        }
        .story-box mark {
            background: #f7d8c2;
            padding: 0 2px;
            border-radius: 2px;
        }
        .story-box mark.diff {
            background: #f7e4be;
        }
        .story-box mark.both {
            background: #f1d8a4;
        }
        .replacement-original {
            color: #1f1f1f;
            font-family: "Georgia", "Times New Roman", serif;
            font-style: italic;
            background: #f7d8c2;
            box-shadow: inset 0 -1px 0 #dfb89d;
            border-radius: 2px;
            padding: 0 2px;
        }
        .transform-cell-spacer {
            height: 74px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            line-height: 1.1;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            white-space: normal !important;
            line-height: 1.2;
            overflow-wrap: anywhere;
            word-break: break-word;
            min-height: 2.1em;
        }
        div[data-testid="stMetricLabel"] > div {
            white-space: normal !important;
        }
        div[data-testid="stMetricLabel"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: unset !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
            display: block;
        }
        div[data-testid="stMetricLabel"] p {
            white-space: pre-line !important;
            overflow: visible !important;
            text-overflow: clip !important;
            margin: 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--su-border);
            border-radius: 12px;
            background: var(--box-bg);
            box-shadow: var(--box-shadow);
            padding: 8px 10px;
        }
        .metric-card {
            border: 1px solid var(--su-border);
            border-radius: 12px;
            background: var(--box-bg);
            box-shadow: var(--box-shadow);
            padding: 9px 16px;
            min-height: 104px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .metric-card .metric-label {
            font-size: 0.78rem;
            line-height: 1.2;
            color: #1f1f1f;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .metric-card .metric-value {
            font-size: 1.55rem;
            line-height: 1.1;
            font-weight: 600;
            color: #1f1f1f;
        }
        div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0.35rem;
        }
        .uni-logo-strip {
            position: fixed;
            left: 16px;
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
            height: 26px;
            width: auto;
            object-fit: contain;
            display: block;
        }
        h1 {
            text-align: center;
            font-size: clamp(1.9rem, 2.9vw, 2.6rem) !important;
            line-height: 1.15;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Non-destructive setup: ensure files exist but preserve content.
    ensure_additional_survey_files_exist()
    model, provider_name = require_startup_config()

    if "hide_transformed_text" not in st.session_state:
        st.session_state.hide_transformed_text = False
    if "active_story_path" not in st.session_state:
        st.session_state.active_story_path = str(INPUT_STORY_PATH)
    if "active_story_name" not in st.session_state:
        st.session_state.active_story_name = INPUT_STORY_PATH.stem

    saved_additional = st.session_state.get("additional_questions", "")
    st.title("NARRATE: Neurosymbolic Abductive Reasoning for Reframing Texts")
    col1, col2, col3 = st.columns([2, 2, 1.5], gap="small")

    with col1:
        top_left, top_right = st.columns([1.4, 1], gap="small")
        with top_right:
            st.markdown(
                (
                    f"<div style='font-size: 0.85rem; color: #4a4a4a;'>"
                    f"Configured Model: <b>{html.escape(model)}</b><br>"
                    f"Provider: <code>{html.escape(provider_name)}</code>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            scenario = st.selectbox("Scenario", SCENARIO_OPTIONS)
            if st.button("Change Setup", use_container_width=True):
                for key in [
                    "_setup_complete",
                    "_setup_model",
                    "_setup_provider_api_key",
                    "_setup_hf_token",
                    "_setup_model_draft",
                    "_setup_provider_key_draft",
                    "_setup_hf_token_draft",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()
        clear_outputs_on_selection_change(model, scenario)
        with top_left:
            story_upload = st.file_uploader(
                "Upload Input Story",
                type=["txt"],
                key="story_file",
                label_visibility="collapsed",
            )
            active_story_path = Path(st.session_state.get("active_story_path", str(INPUT_STORY_PATH)))
            if story_upload is not None:
                uploaded_bytes = story_upload.getvalue()
                uploaded_signature = f"{story_upload.name}:{hashlib.sha1(uploaded_bytes).hexdigest()}"
                uploaded_story_name = sanitize_story_name(Path(story_upload.name).stem)
                active_story_path = INPUT_DIR / f"{uploaded_story_name}.txt"
                st.session_state.active_story_path = str(active_story_path)
                st.session_state.active_story_name = uploaded_story_name
                if st.session_state.get("_last_story_upload_signature") != uploaded_signature:
                    uploaded_text = uploaded_bytes.decode("utf-8", errors="ignore").rstrip("\n")
                    story_text = f"{uploaded_text}\n" if uploaded_text else ""
                    save_text(active_story_path, story_text)
                    # Keep input.txt as a compatibility mirror for existing scripts/workflows.
                    save_text(INPUT_STORY_PATH, story_text)
                    st.session_state.original_story_editor = uploaded_text
                    st.session_state._editor_story_name = uploaded_story_name
                    st.session_state._last_story_upload_signature = uploaded_signature
            else:
                if not active_story_path.exists():
                    active_story_path = INPUT_STORY_PATH
                    st.session_state.active_story_path = str(active_story_path)
                    st.session_state.active_story_name = INPUT_STORY_PATH.stem
        active_story_path = Path(st.session_state.get("active_story_path", str(INPUT_STORY_PATH)))
        active_story_name = sanitize_story_name(
            st.session_state.get("active_story_name", active_story_path.stem)
        )
        current_story_text = load_text(active_story_path, "").rstrip("\n")
        if st.session_state.get("_editor_story_name") != active_story_name:
            st.session_state.original_story_editor = current_story_text
            st.session_state._editor_story_name = active_story_name
        original_col, action_col = st.columns([3.7, 1.3], gap="small")
        with original_col:
            st.markdown(
                "<div style='font-size: 0.95rem; font-weight: 600;'>Original Text</div>",
                unsafe_allow_html=True,
            )
            edited_story_text = st.text_area(
                "Original Text Editor",
                key="original_story_editor",
                height=TOP_BOX_HEIGHT,
                label_visibility="collapsed",
            )
            edited_story_text = (edited_story_text or "").rstrip("\n")
            serialized_story_text = f"{edited_story_text}\n" if edited_story_text else ""
            if load_text(active_story_path, "") != serialized_story_text:
                save_text(active_story_path, serialized_story_text)
                # Keep input.txt synced for compatibility with existing scripts/workflows.
                save_text(INPUT_STORY_PATH, serialized_story_text)
        with action_col:
            st.markdown("<div class='transform-cell-spacer'></div>", unsafe_allow_html=True)
            send_clicked = st.button("**Reframe**", use_container_width=True)
        selected_paths = selection_paths(model, scenario)

        rules_text = load_text(selected_paths["pyreason_rules"], "")

        rule_lines = [line.strip() for line in rules_text.splitlines() if line.strip()]
        rule_options = {f"Rule {idx:03d}: {line}": line for idx, line in enumerate(rule_lines, 1)}
        selected_rule_labels = st.multiselect(
            "Select Rules To Consider",
            options=list(rule_options.keys()),
            default=list(rule_options.keys()),
        )
        selected_rules = [rule_options[label] for label in selected_rule_labels]
        selected_rules_text = "\n".join(selected_rules)
        st.markdown(
            "<div style='font-size: 0.95rem; font-weight: 600;'>Selected Rules</div>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "Selected Rules",
            value=selected_rules_text,
            height=90,
            disabled=True,
            label_visibility="collapsed",
        )

        if selected_rules_text:
            save_text(selected_paths["selected_rules"], selected_rules_text + "\n")
        else:
            save_text(selected_paths["selected_rules"], "")

    try:
        input_story_mtime = active_story_path.stat().st_mtime
        input_story_mtime_ns = active_story_path.stat().st_mtime_ns
    except OSError:
        input_story_mtime = None
        input_story_mtime_ns = None

    questions_for_display = display_questions_for_scenario(scenario)
    original_survey_json = find_phase2_iteration_survey(
        model,
        scenario,
        0,
        story_name=active_story_name,
    )
    optimal_iteration = find_optimal_iteration_for_dashboard(
        model,
        scenario,
        max_scan=PHASE2_ANALYSIS_ITERATIONS,
        story_name=active_story_name,
    )
    transformed_survey_json = find_phase2_iteration_survey(
        model,
        scenario,
        optimal_iteration,
        story_name=active_story_name,
    )
    current_input_story = load_text(active_story_path, "").strip()
    surveys_available = bool(current_input_story and original_survey_json and transformed_survey_json)
    original_story_score = (
        median_from_survey_json(original_survey_json) if surveys_available else "N/A"
    )
    transformed_story_score = (
        median_from_survey_json(transformed_survey_json) if surveys_available else "N/A"
    )
    extended_scores = st.session_state.get("extended_survey_scores")
    if (
        surveys_available
        and isinstance(extended_scores, dict)
        and extended_scores.get("model") == model
        and extended_scores.get("scenario") == scenario
        and extended_scores.get("story_name") == active_story_name
        and extended_scores.get("story_mtime_ns") == input_story_mtime_ns
    ):
        original_story_score = extended_scores.get("original_score", original_story_score)
        transformed_story_score = extended_scores.get("transformed_score", transformed_story_score)
    original_abduction_similarity_path = (
        find_phase2_iteration_file(
            model,
            scenario,
            0,
            "abduction_analysis.json",
            story_name=active_story_name,
        )
        if current_input_story
        else None
    )
    original_scenario_similarity_value = avg_current_confidence_from_abduction(
        original_abduction_similarity_path
    )
    original_scenario_similarity_score = (
        f"{original_scenario_similarity_value:.3f}"
        if original_scenario_similarity_value is not None
        else "N/A"
    )
    abduction_similarity_iteration = optimal_iteration
    abduction_similarity_path = (
        find_phase2_iteration_file(
            model,
            scenario,
            abduction_similarity_iteration,
            "abduction_analysis.json",
            story_name=active_story_name,
        )
        if current_input_story
        else None
    )
    scenario_similarity_value = avg_current_confidence_from_abduction(abduction_similarity_path)
    scenario_similarity_score = (
        f"{scenario_similarity_value:.3f}" if scenario_similarity_value is not None else "N/A"
    )

    with col2:
        current_story_text = load_text(active_story_path, "")
        if send_clicked and current_story_text.strip():
            clear_phase2_output_for_selection(
                model,
                scenario,
                story_name=active_story_name,
            )
            save_text(active_story_path, current_story_text.strip() + "\n")
            save_text(INPUT_STORY_PATH, current_story_text.strip() + "\n")
            rules_input_path = selected_paths["selected_rules"] if selected_rules_text else selected_paths["pyreason_rules"]
            ok, response = run_phase2_transform(
                model,
                scenario,
                rules_path=rules_input_path,
                story_path=active_story_path,
            )
            if ok:
                st.session_state.hide_transformed_text = False
                st.rerun()
            st.error(response)

        st.markdown(
            "<div style='font-size: 0.95rem; font-weight: 600;'>Reframed Text</div>",
            unsafe_allow_html=True,
        )
        ranked_iter_optimal = find_phase2_iteration_file(
            model,
            scenario,
            optimal_iteration,
            "ranked_prescriptions.json",
            story_name=active_story_name,
        )
        original_text = load_text(active_story_path, "").strip()
        if original_text:
            highlight_segments_transformed = segments_from_ranked_prescriptions(ranked_iter_optimal)
            transformed_text = transformed_story_text(
                model,
                scenario,
                optimal_iteration,
                story_name=active_story_name,
            )
        else:
            highlight_segments_transformed = []
            transformed_text = ""
        if st.session_state.hide_transformed_text:
            transformed_html = transformed_with_original_replacements_html(original_text, transformed_text)
        else:
            transformed_html = highlighted_story_html(
                transformed_text,
                highlight_segments_transformed,
                box_class="transformed-box",
            )
        st.markdown(transformed_html, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        hide_left, hide_mid, hide_right = st.columns([1, 1.2, 1])
        with hide_mid:
            if st.button("Hide Changes", key="hide_transform_text", use_container_width=True):
                st.session_state.hide_transformed_text = not st.session_state.hide_transformed_text
                st.rerun()

    with col3:
        score1, score2, score3 = st.columns(3, gap="small")
        original_score_display = format_score_integer(original_story_score)
        transformed_score_display = format_score_integer(transformed_story_score)
        semantic_similarity_value = semantic_similarity_from_cache_or_compute(
            model=model,
            scenario=scenario,
            story_name=active_story_name,
            iteration=optimal_iteration,
            original_text=original_text,
            transformed_text=transformed_text,
        )
        semantic_similarity_display = (
            f"{semantic_similarity_value:.3f}" if semantic_similarity_value is not None else "N/A"
        )
        original_similarity_score_display = original_scenario_similarity_score
        reframed_similarity_score_display = scenario_similarity_score
        score1.markdown(
            f"<div class='metric-card'><div class='metric-label'>Original Survey<br>Score (1-5)</div><div class='metric-value'>{original_score_display}</div></div>",
            unsafe_allow_html=True,
        )
        score2.markdown(
            f"<div class='metric-card'><div class='metric-label'>Reframed Text<br>Score (1-5)</div><div class='metric-value'>{transformed_score_display}</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("Note: Survey score scale is 1 = individualistic and 5 = collectivistic.")
        score3.markdown(
            f"<div class='metric-card'><div class='metric-label'>Semantic Similarity<br>(0-1)</div><div class='metric-value'>{semantic_similarity_display}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        score4, score5 = st.columns(2, gap="small")
        score4.markdown(
            f"<div class='metric-card'><div class='metric-label'>Reframed Scenario<br>Similarity Score (0-1)</div><div class='metric-value'>{reframed_similarity_score_display}</div></div>",
            unsafe_allow_html=True,
        )
        score5.markdown(
            f"<div class='metric-card'><div class='metric-label'>Original Scenario<br>Similarity Score (0-1)</div><div class='metric-value'>{original_similarity_score_display}</div></div>",
            unsafe_allow_html=True,
        )

        questions_text = "\n\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(questions_for_display)
        ) if questions_for_display else "No survey questions found in JSON."
        st.markdown(
            "<div style='font-size: 0.95rem; font-weight: 600;'>Survey</div>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "Survey",
            value=questions_text,
            height=SURVEY_BOX_HEIGHT,
            disabled=True,
            label_visibility="collapsed",
        )

        st.markdown(
            "<div style='font-size: 0.9rem; font-weight: 600; margin-bottom: 0.2rem;'>Add Survey Questions</div>",
            unsafe_allow_html=True,
        )
        additional_q = st.text_area(
            "Add Survey Questions",
            value=saved_additional,
            height=ADDITIONAL_SURVEY_BOX_HEIGHT,
            key="additional_questions",
            placeholder="Intended narrative characteristic.\nSurvey question?",
            label_visibility="collapsed",
        )
        if st.button("Extend Survey", use_container_width=True):
            additional_survey_path = additional_survey_path_for_scenario(scenario)
            added_count = append_additional_survey_json(additional_q, scenario, additional_survey_path)
            if added_count > 0:
                if transformed_text.strip() and transformed_text != "No transformed story found.":
                    try:
                        recomputed_original, recomputed_transformed, run_cost = recompute_scores_with_extended_survey(
                            model=model,
                            scenario=scenario,
                            original_text=original_text,
                            transformed_text=transformed_text,
                        )
                        st.session_state["extended_survey_scores"] = {
                            "model": model,
                            "scenario": scenario,
                            "story_name": active_story_name,
                            "story_mtime_ns": input_story_mtime_ns,
                            "original_score": recomputed_original,
                            "transformed_score": recomputed_transformed,
                            "cost": run_cost,
                        }
                        st.success(
                            f"Appended {added_count} question block(s) to {additional_survey_path.relative_to(BASE_DIR)} "
                            f"and recomputed scores (cost: ${run_cost:.4f})."
                        )
                    except Exception as e:
                        st.warning(
                            f"Questions were saved, but score recomputation failed: {e}"
                        )
                else:
                    st.warning(
                        "Questions were saved, but no transformed story was found to recompute scores."
                    )
                st.rerun()
            else:
                st.warning("No non-empty questions to append.")

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
    leibniz_fallback_svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='220' height='70' viewBox='0 0 220 70'>
      <rect width='220' height='70' rx='10' fill='#1C3B6B'/>
      <text x='110' y='42' text-anchor='middle' font-size='22' font-family='Arial, sans-serif' fill='white' font-weight='700'>LEIBNIZ</text>
    </svg>
    """
    syracuse_logo = resolve_logo_src("syracuse_logo.png", svg_to_data_uri(syracuse_fallback_svg))
    asu_logo = resolve_logo_src("asu_logo.png", svg_to_data_uri(asu_fallback_svg))
    leibniz_logo = resolve_logo_src("leibniz.jpeg", svg_to_data_uri(leibniz_fallback_svg))
    st.markdown(
        f"""
        <div class="uni-logo-strip">
            <img src="{syracuse_logo}" alt="Syracuse University logo" />
            <img src="{asu_logo}" alt="Arizona State University logo" />
            <img src="{leibniz_logo}" alt="Leibniz logo" />
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
