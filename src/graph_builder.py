"""
Graph Builder Module

Converts LLM survey results into PyReason ground atoms format.
Extracts feature ratings and creates the knowledge graph structure needed for reasoning.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ==========================================================
# Rating Normalization
# ==========================================================

def normalize_rating(rating: int) -> float:
    """
    Convert rating (1-5) to normalized score (0.0-1.0).

    Mapping (consistent with Phase 1):
        1 (entirely individual)  → 1.0
        2 (primarily individual) → 0.75
        3 (balanced)             → 0.5
        4 (primarily group)      → 0.25
        5 (entirely group)       → 0.0

    Args:
        rating: Rating value (1-5)

    Returns:
        Normalized score (0.0-1.0)
    """
    mapping = {
        1: 1.0,
        2: 0.75,
        3: 0.5,
        4: 0.25,
        5: 0.0
    }
    return mapping.get(rating, 0.5)  # Default to 0.5 if unknown


# ==========================================================
# Name Conversion Utilities
# ==========================================================

def to_atom_id(name: str) -> str:
    """
    Convert story name to atom ID format.

    Examples:
        "Test Story" → "test_story"
        "Jay Z" → "jay_z"

    Args:
        name: Story name

    Returns:
        Atom ID (lowercase, underscores)
    """
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def component_to_predicate(component: str) -> str:
    """
    Convert component name to predicate format.

    Examples:
        "Protagonist‑Centered Focus" → "protagonist_centered_focus"
        "Internal Goals" → "internal_goals"
        "Man vs. Self/World" → "man_vs_self_world"

    Args:
        component: Feature component name

    Returns:
        Predicate format (lowercase, underscores)
    """
    # First, replace all non-alphanumeric characters (except spaces) with underscores
    # This handles: ‑, -, /, &, etc.
    cleaned = re.sub(r'[^\w\s]', '_', component)

    # Replace multiple spaces/underscores with single underscore
    cleaned = re.sub(r'[\s_]+', '_', cleaned)

    # Lowercase and strip leading/trailing underscores
    return cleaned.lower().strip('_')
#
#
# def component_to_constant(component: str) -> str:
#     """
#     Convert component name to constant format (for ground atoms).
#
#     Examples:
#         "Protagonist-Centered Focus" → "protagonist centered focus"
#         "Internal Goals" → "internal goals"
#
#     Args:
#         component: Feature component name
#
#     Returns:
#         Constant format (lowercase, spaces)
#     """
#     # Remove non-alphanumeric except spaces
#     cleaned = re.sub(r'[^\w\s]', '', component)
#     # Normalize spaces
#     return re.sub(r'\s+', ' ', cleaned.strip()).lower()


# ==========================================================
# Extract Rating from LLM Response
# ==========================================================

def extract_rating_from_response(llm_response: str) -> Optional[int]:
    """
    Extract rating (1-5) from LLM response.

    Expected formats:
        "Rating: 1"
        "1"
        "The rating is 2..."

    Args:
        llm_response: Raw LLM response text

    Returns:
        Rating (1-5) or None if not found
    """
    if not llm_response:
        return None

    # Try first line
    first_line = llm_response.split('\n')[0].strip()

    # Check if starts with "Rating:"
    if first_line.lower().startswith('rating:'):
        rating_str = first_line.split(':', 1)[1].strip()
    else:
        rating_str = first_line

    # Extract digit 1-5
    match = re.search(r'\b([1-5])\b', rating_str)
    if match:
        return int(match.group(1))

    return None


def parse_llm_excerpts(llm_response: str) -> List[str]:
    """
    Parse excerpts/quotes from LLM response.

    Looks for numbered excerpts with quotes: '1. "text"'
    This matches the survey prompt format where LLM is asked to
    "quote a few short excerpts from the story".

    Args:
        llm_response: Raw LLM response text

    Returns:
        List of excerpt strings
    """
    excerpts = []
    lines = llm_response.split('\n')

    for line in lines:
        line = line.strip()

        # Look for numbered excerpts with quotes: "1. \"text\""
        # Handles various quote types: ", ", ', '
        numbered_match = re.match(r'^\d+\.\s*[""\'"]([^""\'\"]+)[""\'"]', line)
        if numbered_match:
            excerpt_text = numbered_match.group(1).strip()
            if excerpt_text:
                excerpts.append(excerpt_text)

    return excerpts


# ==========================================================
# Main Graph Building Function
# ==========================================================

def build_ground_atoms_from_survey(
        survey_json_path: str,
        story_name: str
) -> Tuple[Dict[str, List[float]], Dict[str, Dict]]:
    """
    Build ground atoms and segments metadata from survey JSON.

    Args:
        survey_json_path: Path to survey JSON file
        story_name: Name of the story (for atom ID)

    Returns:
        Tuple of:
            - ground_atoms: Dict mapping atom strings to [lower, upper] bounds
            - segments_metadata: Dict mapping features to their metadata
    """
    logger.info(f"Building ground atoms from survey: {survey_json_path}")

    # Load survey JSON
    with open(survey_json_path, 'r', encoding='utf-8') as f:
        survey_data = json.load(f)

    story_id = to_atom_id(story_name)
    ground_atoms: Dict[str, List[float]] = {}
    segments_metadata: Dict[str, Dict] = {}

    # Process each question-answer pair
    qa_list = survey_data.get('questions_and_answers', [])

    for qa in qa_list:
        # Only process individualistic questions (Phase 2 uses same format)
        # if qa.get('type') != 'individualistic_question':
        #     continue

        component = qa.get('component', '')
        if not component:
            logger.warning("Question missing component, skipping")
            continue

        llm_response = qa.get('raw_response', '')
        question_text = qa.get('question', '')

        # Extract rating
        rating = qa.get('rating')  # May be stored directly
        if rating is None:
            rating = extract_rating_from_response(llm_response)

        if rating is None:
            logger.warning(f"Could not extract rating for component: {component}")
            continue

        # Normalize rating to score
        normalized_score = normalize_rating(rating)

        # Convert component to predicate format (use underscores consistently)
        feature_predicate = component_to_predicate(component)

        # Create ground atoms (use same format as learned rules)
        # Format: individualistic_feature(story_id, feature_predicate):[score, score]
        ground_atoms[f'individualistic_feature({story_id}, {feature_predicate})'] = [
            normalized_score, normalized_score
        ]

        # Add supporting atoms
        ground_atoms[f'story_name({story_id})'] = [1.0, 1.0]
        ground_atoms[f'{feature_predicate}({feature_predicate})'] = [1.0, 1.0]

        # Extract segments/excerpts
        excerpts = parse_llm_excerpts(llm_response)

        # Store metadata for this feature
        segments_metadata[feature_predicate] = {
            'rating': rating,
            'normalized_score': normalized_score,
            'llm_response': llm_response,
            'question': question_text,
            'excerpts': excerpts,
            'component': component  # Original component name
        }

    logger.info(f"Created {len(ground_atoms)} ground atoms")
    logger.info(f"Extracted metadata for {len(segments_metadata)} features")

    return ground_atoms, segments_metadata


def save_ground_atoms(ground_atoms: Dict[str, List[float]], output_path: str):
    """
    Save ground atoms to JSON file.

    Args:
        ground_atoms: Ground atoms dictionary
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_atoms, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved ground atoms to: {output_path}")


def save_segments_metadata(segments_metadata: Dict, output_path: str):
    """
    Save segments metadata to JSON file.

    Args:
        segments_metadata: Segments metadata dictionary
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Wrap in story name for compatibility with existing scripts
    # (They expect: {story_name: {feature: metadata}})
    wrapped = {"story": segments_metadata}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(wrapped, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved segments metadata to: {output_path}")


# ==========================================================
# For Testing/Direct Execution
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python graph_builder.py <survey_json> <story_name>")
        print(
            "Example: python graph_builder.py output/phase2/gpt-4o/forward/test_story/iteration_0/survey.json 'Test Story'")
        sys.exit(1)

    survey_json = sys.argv[1]
    story_name = sys.argv[2]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Build ground atoms
    ground_atoms, segments_metadata = build_ground_atoms_from_survey(
        survey_json, story_name
    )

    # Save to current directory (for testing)
    save_ground_atoms(ground_atoms, "ground_atoms.json")
    save_segments_metadata(segments_metadata, "segments_metadata.json")

    print(f"\n✓ Ground atoms created: {len(ground_atoms)} atoms")
    print(f"✓ Segments metadata: {len(segments_metadata)} features")
    print("\nSample ground atoms:")
    for i, (atom, bounds) in enumerate(list(ground_atoms.items())[:5], 1):
        print(f"  {i}. {atom}: {bounds}")