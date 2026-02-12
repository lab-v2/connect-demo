"""
LLM Survey Module

Conducts surveys on stories using various LLM providers.
Supports: OpenAI, Anthropic, OpenRouter, xAI/Grok, Gemini, AWS Bedrock

Features:
- Multi-provider support via LiteLLM
- Retry with exponential backoff for rate limits
- Cost and token tracking
- Progress tracking with tqdm
- Intermediate result saving after each story
"""

import os
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from decimal import Decimal



try:
    from litellm import completion
    from litellm.exceptions import RateLimitError, APIError
except ImportError:
    raise ImportError(
        "litellm is required. Install with: pip install litellm"
    )

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "tqdm is required for progress bars. Install with: pip install tqdm"
    )

logger = logging.getLogger(__name__)

# ==========================================================
# Cost Tracking
# ==========================================================

# Pricing per 1K tokens (adjust as needed)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-5.2": {"input": 0.00175, "output": 0.014},
    "gpt-4o-mini-2024-07-18": {"input": 0.0008, "output": 0.0032},
    "ft:gpt-4o-mini-2024-07-18:syracuse-university:llm2:D5JJuHZi": {"input": 0.0008, "output": 0.0032},
    "xai/grok-4-fast-reasoning": {"input": 0.0002, "output": 0.0005},
    # Anthropic Claude
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},  # $3/$15 per 1M = $0.003/$0.015 per 1K

    # AWS Bedrock - Meta Llama
    "bedrock/us.meta.llama4-maverick-17b-instruct-v1:0": {"input": 0.00024, "output": 0.00097},
    "meta.llama4": {"input": 0.00024, "output": 0.00097},  # Fallback
    "bedrock/us.meta.llama3-2-11b-instruct-v1:0": {"input": 0.00016, "output": 0.00072},  # Fallback
    # AWS Bedrock - DeepSeek
    "bedrock/us.deepseek.r1-v1:0": {"input": 0.00135, "output": 0.0054},
    "bedrock/deepseek-llm-r1-distill-qwen-32b": {"input": 0.00135, "output": 0.0054}


    # # Anthropic
    # "claude-3-opus": {"input": 0.015, "output": 0.075},
    # "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    # "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    #
    # # OpenRouter (placeholder - varies by model)
    # "openrouter": {"input": 0.002, "output": 0.006},
    #
    # # xAI Grok
    # "xai/grok": {"input": 0.005, "output": 0.015},
    #
    # # Gemini
    # "gemini": {"input": 0.00025, "output": 0.00125},
    #
    # # AWS Bedrock (placeholder)
    # "bedrock": {"input": 0.003, "output": 0.015},
}


def get_pricing_for_model(model: str) -> Dict[str, float]:
    """
    Get pricing for a model.

    Args:
        model: Model name

    Returns:
        Dict with 'input' and 'output' price per 1K tokens
    """
    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try prefix matching
    for key in MODEL_PRICING:
        if model.startswith(key):
            return MODEL_PRICING[key]

    # Default fallback
    logger.warning(f"No pricing found for model {model}, using default")
    return {"input": 0.0025, "output": 0.01}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for a model call.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = get_pricing_for_model(model)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return input_cost + output_cost


class CostTracker:
    """Tracks tokens and costs across LLM calls."""

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.per_story_breakdown = []

    def add_call(self, story_name: str, input_tokens: int, output_tokens: int):
        """Add a call to the tracker."""
        cost = calculate_cost(self.model, input_tokens, output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        self.per_story_breakdown.append({
            "story_name": story_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "timestamp": datetime.now().isoformat()
        })

    def to_dict(self) -> Dict:
        """Convert tracker to dictionary for JSON serialization."""
        return {
            "model": self.model,
            "total_stories_processed": len(self.per_story_breakdown),
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens
            },
            "total_cost_usd": round(self.total_cost, 6),
            "per_story_breakdown": self.per_story_breakdown
        }

    def save(self, output_path: Path):
        """Save cost tracking to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"Cost tracking saved to {output_path}")


# ==========================================================
# API Key Management
# ==========================================================

def get_api_key_for_model(model: str) -> Optional[str]:
    """
    Get the appropriate API key based on model prefix.

    Args:
        model: Model name

    Returns:
        API key string, or None for AWS Bedrock models

    Raises:
        RuntimeError: If required API key is not set
    """
    if model.startswith("openrouter/"):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Set it with: export OPENROUTER_API_KEY='your-key'"
            )
        return key

    if model.startswith("xai/"):
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise RuntimeError(
                "XAI_API_KEY environment variable not set. "
                "Set it with: export XAI_API_KEY='your-key'"
            )
        return key

    if model.startswith("anthropic/") or model.startswith("claude"):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key'"
            )
        return key

    # if model.startswith("gemini/"):
    #     key = os.getenv("GEMINI_API_KEY")
    #     if not key:
    #         raise RuntimeError(
    #             "GEMINI_API_KEY environment variable not set. "
    #             "Set it with: export GEMINI_API_KEY='your-key'"
    #         )
    #     return key

    # AWS Bedrock models (qwen, bedrock/)
    if model.startswith("bedrock/") or model.startswith("qwen"):
        # Use AWS credentials (no explicit API key)
        return None

    # Default to OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key'"
        )
    return key


# ==========================================================
# LLM Calling with Retry Logic
# ==========================================================

def call_llm_with_retry(
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3
) -> Tuple[str, int, int]:
    """
    Call LLM with exponential backoff retry logic.

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        max_retries: Maximum number of retries

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)

    Raises:
        Exception: If all retries fail
    """
    api_key = get_api_key_for_model(model)

    # OpenRouter specific headers
    extra_headers = None
    if model.startswith("openrouter/"):
        extra_headers = {
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "CONNECT Project"
        }

    for attempt in range(max_retries):
        try:
            # Build kwargs
            # GPT-5.x models use max_completion_tokens instead of max_tokens
            if model.startswith("gpt-5"):
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_completion_tokens": max_tokens,  # New parameter for GPT-5.x
                    "temperature": temperature
                }
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": max_tokens,  # Old parameter for other models
                    "temperature": temperature
                }

            # Add API key if not Bedrock
            if api_key is not None:
                kwargs["api_key"] = api_key

            # Add extra headers if OpenRouter
            if extra_headers is not None:
                kwargs["extra_headers"] = extra_headers

            # Make the call
            response = completion(**kwargs)

            # Extract response and usage
            response_text = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {}) or {}
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))

            return response_text, input_tokens, output_tokens

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts")
                raise

        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"API error after {max_retries} attempts: {e}")
                raise

        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            raise

    raise RuntimeError("Should not reach here")


# ==========================================================
# Response Parsing
# ==========================================================

def parse_llm_response(response_text: str) -> Tuple[Optional[int], str, str]:
    """
    Parse LLM response to extract rating, justification, and evidence.

    Expected format:
        Rating: 1

        Justification text here...

        Evidence: "Quote from story..."

    Args:
        response_text: Raw LLM response

    Returns:
        Tuple of (rating, justification, evidence)
    """
    # Try to extract rating
    rating = None
    rating_match = re.search(r'(?:Rating:|^)\s*(\d)', response_text, re.MULTILINE | re.IGNORECASE)
    if rating_match:
        rating = int(rating_match.group(1))
    else:
        # Fallback: try to find any digit at the start
        first_line_match = re.match(r'\s*(\d)', response_text)
        if first_line_match:
            rating = int(first_line_match.group(1))

    # Try to extract evidence (text in quotes)
    evidence = ""
    evidence_matches = re.findall(r'"([^"]+)"', response_text)
    if evidence_matches:
        evidence = " | ".join(evidence_matches[:3])  # Take first 3 quotes

    # Justification is everything (we keep full response for now)
    justification = response_text.strip()

    return rating, justification, evidence


# ==========================================================
# Survey Functions
# ==========================================================

def load_questions(questions_file: str) -> List[Dict]:
    """
    Load questions from JSON file.

    Args:
        questions_file: Path to questions JSON file

    Returns:
        List of question dictionaries
    """
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["questions"]


def conduct_survey_single_story(
        story: Dict[str, str],
        questions_file: str,
        problem_type: str,
        model: str,
        temperature: float = 0.0,
        cost_tracker: Optional[CostTracker] = None
) -> Dict:
    """
    Conduct LLM survey on a single story.

    Args:
        story: Story dict with 'name', 'path', 'content'
        questions_file: Path to questions JSON file
        problem_type: "forward" or "inverse"
        model: Model name
        temperature: LLM temperature
        cost_tracker: Optional cost tracker to update

    Returns:
        Survey result dictionary
    """
    questions = load_questions(questions_file)

    # Determine which question type to use
    question_type = "individualistic_question" if problem_type == "forward" else "collectivistic_question"

    result = {
        "story_name": story["name"],
        "story_file": story["path"],
        "model": model,
        "problem_type": problem_type,
        "timestamp": datetime.now().isoformat(),
        "questions_and_answers": []
    }

    story_total_input = 0
    story_total_output = 0

    # Process each question
    for q in questions:
        if question_type not in q:
            logger.warning(f"Question {q['id']} missing {question_type}, skipping")
            continue

        question_obj = q[question_type]
        question_text = question_obj["question"]
        rating_instructions = question_obj["rating"]

        # Format prompts (matching their style)
        system_prompt = "\n".join(rating_instructions)
        user_prompt = (
            f"STORY:\n{story['content']}\n\n"
            f"QUESTION:\n{question_text}\n\n"
            "Use the following 1–5 rating scale:\n"
            "1 = Entirely individual perspective\n"
            "2 = Primarily individual but with some group influence\n"
            "3 = Balanced between individual and group\n"
            "4 = Primarily group-oriented\n"
            "5 = Entirely group/community perspective\n\n"
            "Answer the question about the story using the scale and briefly justify your answer. "
            "Put the rating FIRST (e.g., 'Rating: 4'), THEN justify. "
            "Finally, quote a few short excerpts from the story that exemplify your chosen rating."
        )
        # Add explicit format instruction for non-GPT models
        # if not model.startswith("gpt-4"):
        user_prompt += (
            " Format excerpts as a simple numbered list with plain quotes:\n"
            "1. \"excerpt text\"\n"
            "2. \"excerpt text\"\n"
            "Do NOT use bold (**), italics (*), or explanatory notes in parentheses."
        )
        try:
            # Call LLM with retry logic
            response_text, input_tokens, output_tokens = call_llm_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=4096,
                max_retries=3
            )

            story_total_input += input_tokens
            story_total_output += output_tokens

            # Parse response
            rating, justification, evidence = parse_llm_response(response_text)

            # Store result
            entry = {
                "id": q["id"],
                "component": q["component"],
                "type": question_type,
                "question": question_text,
                "rating": rating,
                "justification": justification,
                "evidence": evidence,
                "raw_response": response_text,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens
                }
            }
            result["questions_and_answers"].append(entry)

        except Exception as e:
            logger.error(f"Failed to process question {q['id']} for story {story['name']}: {e}")
            # Store error entry
            entry = {
                "id": q["id"],
                "component": q["component"],
                "type": question_type,
                "question": question_text,
                "error": str(e),
                "rating": None,
                "raw_response": None
            }
            result["questions_and_answers"].append(entry)

    # Update cost tracker
    if cost_tracker:
        cost_tracker.add_call(story["name"], story_total_input, story_total_output)

    # Add token summary to result
    result["total_tokens"] = {
        "input": story_total_input,
        "output": story_total_output,
        "total": story_total_input + story_total_output
    }

    return result


def sanitize_model_name(model: str) -> str:
    """
    Sanitize model name for use in directory paths.

    Args:
        model: Model name (e.g., "openrouter/meta-llama/llama-4")

    Returns:
        Sanitized name (e.g., "openrouter-meta-llama-llama-4")
    """
    return model.replace("/", "-").replace(":", "-")


def conduct_surveys(
        stories: List[Dict[str, str]],
        questions_file: str,
        problem_type: str,
        model: str,
        temperature: float,
        output_dir: str
) -> Tuple[List[Dict], List[str]]:
    """
    Conduct surveys on multiple stories.

    Args:
        stories: List of story dicts from story_loader
        questions_file: Path to questions JSON file
        problem_type: "forward" or "inverse"
        model: Model name
        temperature: LLM temperature
        output_dir: Base output directory

    Returns:
        Tuple of (list of survey results, list of failed story names)
    """
    # Create model-specific directory
    sanitized_model = sanitize_model_name(model)
    model_dir = Path(output_dir) / sanitized_model / problem_type
    survey_dir = model_dir / "survey_results"
    survey_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Survey results will be saved to: {survey_dir}")

    # Initialize cost tracker
    cost_tracker = CostTracker(model)

    # Track results and failures
    all_results = []
    failed_stories = []

    # Process each story with progress bar
    for story in tqdm(stories, desc=f"Surveying stories ({model})", unit="story"):
        logger.info(f"Processing story: {story['name']}")

        # Check if already processed
        output_file = survey_dir / f"{story['name']}.json"
        if output_file.exists():
            logger.info(f"Story {story['name']} already processed, skipping")
            # Load existing result for completeness
            with open(output_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            all_results.append(result)
            continue

        try:
            # Conduct survey
            result = conduct_survey_single_story(
                story=story,
                questions_file=questions_file,
                problem_type=problem_type,
                model=model,
                temperature=temperature,
                cost_tracker=cost_tracker
            )

            # Save result immediately
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Saved: {output_file.name}")
            all_results.append(result)

            # Save cost tracking after each story
            cost_file = model_dir / "cost_tracking.json"
            cost_tracker.save(cost_file)

        except Exception as e:
            logger.error(f"✗ Failed to process story {story['name']}: {e}")
            failed_stories.append(story['name'])
            continue

    # Save final cost tracking
    cost_file = model_dir / "cost_tracking.json"
    cost_tracker.save(cost_file)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Survey complete!")
    logger.info(f"Processed: {len(all_results)}/{len(stories)} stories")
    logger.info(f"Failed: {len(failed_stories)} stories")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"{'=' * 60}\n")

    # Save failed stories list if any
    if failed_stories:
        failed_file = model_dir / "failed_stories.json"
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "failed_count": len(failed_stories),
                "failed_stories": failed_stories,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        logger.warning(f"Failed stories list saved to: {failed_file}")

    return all_results, failed_stories