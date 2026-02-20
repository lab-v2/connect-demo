"""
Story Transformer Single Module

Transforms stories by rewriting ALL identified segments in a SINGLE LLM call,
instead of iterating segment-by-segment.
Uses LLM to modify all segments at once to shift narrative style.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from llm_survey import call_llm_with_retry, CostTracker

logger = logging.getLogger(__name__)

# ==========================================================
# Prompt Templates
# ==========================================================

COLLECTIVIST = "collectivist"
INDIVIDUALIST = "individualist"

PROMPT_TEMPLATE_SINGLE = '''You are given a {source_type} story below.



{story}



Your goal is to make the story more {target_type}. To do this, you will update ONLY the following selected segments from the story. Each segment is listed below:

{segments_list}

Now, rewrite the whole story by updating ONLY the selected segments above to make the story more {target_type}. Do not change any other part of the story. Just output the rewritten story, nothing else. 
'''


# ==========================================================
# Prescription Processing (same helpers as story_transformer.py)
# ==========================================================

def group_prescriptions_by_feature(prescriptions: List[Dict]) -> Dict[str, List[Dict]]:
    feature_groups = defaultdict(list)
    for presc in prescriptions:
        feature = presc['feature']
        feature_groups[feature].append(presc)
    return feature_groups


def select_top_k_features(
        prescriptions: List[Dict],
        top_k: int,
        problem_type: str
) -> Tuple[List[str], List[Dict]]:
    feature_groups = group_prescriptions_by_feature(prescriptions)

    seen_features = []
    for presc in prescriptions:
        if presc['feature'] not in seen_features:
            if problem_type == "forward":
                if presc['target_rating'] < presc['current_rating']:
                    seen_features.append(presc['feature'])
            elif problem_type == "inverse":
                if presc['target_rating'] > presc['current_rating']:
                    seen_features.append(presc['feature'])

    selected_features = seen_features[:top_k]

    selected_prescriptions = []
    for feature in selected_features:
        selected_prescriptions.extend(feature_groups[feature])

    selected_prescriptions.sort(key=lambda x: x['rank'])

    logger.info(f"Selected top {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        num_segments = len(feature_groups[feature])
        logger.info(f"  {i}. {feature}: {num_segments} segments")

    logger.info(f"Total segments to transform: {len(selected_prescriptions)}")

    return selected_features, selected_prescriptions


def filter_reversals(
        prescriptions: List[Dict],
        problem_type: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter out reversal prescriptions (where direction is wrong).

    Returns:
        Tuple of (valid_prescriptions, skipped_reversals)
    """
    valid = []
    skipped = []

    for presc in prescriptions:
        current_rating = presc['current_rating']
        target_rating = presc['target_rating']

        is_reversal = False
        if problem_type == "forward":
            if target_rating > current_rating:
                is_reversal = True
        else:  # inverse
            if target_rating < current_rating:
                is_reversal = True

        if is_reversal:
            logger.warning(
                f"  ⚠️  SKIPPING REVERSAL: {presc['feature']} "
                f"rating {current_rating} → {target_rating} not allowed in {problem_type} problem"
            )
            skipped.append(presc)
        else:
            valid.append(presc)

    return valid, skipped


# ==========================================================
# Prompt Creation
# ==========================================================

def create_single_transformation_prompt(
        story_text: str,
        prescriptions: List[Dict],
        problem_type: str
) -> str:
    """
    Create a single transformation prompt containing ALL segments to change.

    Args:
        story_text: Current full story text
        prescriptions: List of valid (non-reversal) prescriptions to transform
        problem_type: "forward" or "inverse"

    Returns:
        Formatted prompt string
    """
    if problem_type == "forward":
        source_type = COLLECTIVIST
        target_type = INDIVIDUALIST
    else:
        source_type = INDIVIDUALIST
        target_type = COLLECTIVIST

    # Build numbered segments list
    segments_lines = []
    for i, presc in enumerate(prescriptions, 1):
        segments_lines.append(
            f"  {i}. [Feature: {presc['feature']}] ** {presc['segment_text']} **"
        )
    segments_list = "\n".join(segments_lines)

    prompt = PROMPT_TEMPLATE_SINGLE.format(
        source_type=source_type,
        target_type=target_type,
        story=story_text,
        segments_list=segments_list
    )

    return prompt


# ==========================================================
# Story Transformation (Single Call)
# ==========================================================

def transform_story_single(
        story_text: str,
        prescriptions_file: str,
        top_k: int,
        problem_type: str,
        model: str,
        temperature: float,
        output_dir: str
) -> Tuple[str, str, str]:
    """
    Transform story by rewriting ALL segments in a SINGLE LLM call.

    Args:
        story_text: Original story text
        prescriptions_file: Path to ranked_prescriptions.json
        top_k: Number of top features to use
        problem_type: "forward" or "inverse"
        model: LLM model name
        temperature: LLM temperature
        output_dir: Directory to save outputs (same structure as story_transformer.py)

    Returns:
        Tuple of (final_story, transformation_log_path, cost_tracking_path)
    """
    logger.info("=" * 60)
    logger.info("STORY TRANSFORMATION (SINGLE CALL)")
    logger.info("=" * 60)

    # Load prescriptions
    with open(prescriptions_file, 'r', encoding='utf-8') as f:
        prescriptions_data = json.load(f)

    prescriptions = prescriptions_data.get('prescriptions', [])
    story_name = prescriptions_data.get('story_name', 'unknown')

    if not prescriptions:
        logger.warning("No prescriptions found - nothing to transform")
        return story_text, None, None

    # Select top-k features and their segments
    selected_features, selected_prescriptions = select_top_k_features(
        prescriptions, top_k, problem_type
    )

    # Filter out reversals upfront
    valid_prescriptions, skipped_prescriptions = filter_reversals(
        selected_prescriptions, problem_type
    )

    if not valid_prescriptions:
        logger.warning("All prescriptions were reversals - nothing to transform")
        return story_text, None, None

    logger.info(f"Valid segments for transformation: {len(valid_prescriptions)}")
    logger.info(f"Skipped (reversals): {len(skipped_prescriptions)}")

    # Initialize cost tracker
    cost_tracker = CostTracker(model)

    # Build transformation log
    transformation_log = {
        'story_name': story_name,
        'problem_type': problem_type,
        'model': model,
        'temperature': temperature,
        'k_features': len(selected_features),
        'n_segments': len(selected_prescriptions),
        'n_valid_segments': len(valid_prescriptions),
        'n_skipped_reversals': len(skipped_prescriptions),
        'mode': 'single_call',
        'transformations': []
    }

    # Log skipped reversals
    for presc in skipped_prescriptions:
        transformation_log['transformations'].append({
            'feature': presc['feature'],
            'segment': presc['segment_text'],
            'current_rating': presc['current_rating'],
            'target_rating': presc['target_rating'],
            'status': 'skipped_reversal',
            'reason': f"Rating {presc['current_rating']}→{presc['target_rating']} not allowed in {problem_type} problem"
        })

    # Create single prompt with all valid segments
    prompt = create_single_transformation_prompt(
        story_text=story_text,
        prescriptions=valid_prescriptions,
        problem_type=problem_type
    )

    logger.info(f"\n[Single LLM Call] Transforming {len(valid_prescriptions)} segments at once...")
    for i, presc in enumerate(valid_prescriptions, 1):
        logger.info(f"  {i}. [{presc['feature']}] {presc['segment_text'][:60]}... "
                    f"({presc['current_rating']} → {presc['target_rating']})")

    try:
        # Single LLM call for ALL segments
        rewritten_story, input_tokens, output_tokens = call_llm_with_retry(
            system_prompt="You are a literary expert.",
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=8096,
            max_retries=3
        )

        # Track cost
        cost_tracker.add_call(
            story_name=f"{story_name}_single_call",
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # Log the single transformation call
        transformation_log['transformations'].append({
            'step': 'single_call',
            'segments_transformed': [
                {
                    'feature': presc['feature'],
                    'segment': presc['segment_text'],
                    'current_rating': presc['current_rating'],
                    'target_rating': presc['target_rating'],
                    'confidence_gap': presc['confidence_gap'],
                }
                for presc in valid_prescriptions
            ],
            'prompt_used': prompt,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens
            },
            'cost': round(cost_tracker.per_story_breakdown[-1]['cost_usd'], 6),
            'status': 'success'
        })

        final_story = rewritten_story
        logger.info(f"  ✓ Single-call transformation done (tokens: {input_tokens + output_tokens})")

    except Exception as e:
        logger.error(f"  ✗ Single-call transformation failed: {e}")
        transformation_log['transformations'].append({
            'step': 'single_call',
            'error': str(e),
            'status': 'failed'
        })
        final_story = story_text  # Fall back to original

    # Save final transformed story (same filename as iterative version)
    story_file = Path(output_dir) / "story_transformed.txt"
    story_file.write_text(final_story, encoding='utf-8')
    logger.info(f"\n✓ Final transformed story saved to: {story_file}")

    # Save transformation log (same filename as iterative version)
    log_file = Path(output_dir) / "transformation_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(transformation_log, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Transformation log saved to: {log_file}")

    # Save cost tracking (same filename as iterative version)
    cost_data = cost_tracker.to_dict()
    cost_data['k_features'] = len(selected_features)
    cost_data['n_segments'] = len(selected_prescriptions)
    cost_data['successful_transformations'] = (
        len(valid_prescriptions)
        if any(t.get('status') == 'success' for t in transformation_log['transformations'])
        else 0
    )

    cost_file = Path(output_dir) / "cost_tracking_transformation.json"
    with open(cost_file, 'w', encoding='utf-8') as f:
        json.dump(cost_data, f, indent=2)
    logger.info(f"✓ Transformation costs saved to: {cost_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMATION COMPLETE (SINGLE CALL)")
    logger.info("=" * 60)
    logger.info(f"Features selected: {len(selected_features)}")
    logger.info(f"Segments attempted: {len(selected_prescriptions)}")
    logger.info(f"Segments in single call: {len(valid_prescriptions)}")
    logger.info(f"Segments skipped (reversals): {len(skipped_prescriptions)}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info("=" * 60)

    return final_story, str(log_file), str(cost_file)


# ==========================================================
# For Testing/Direct Execution
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python story_transformer_single.py <story.txt> <prescriptions.json> <top_k> <problem_type> <model>")
        print("Example: python story_transformer_single.py story.txt ranked_prescriptions.json 3 forward gpt-4o")
        sys.exit(1)

    story_file = sys.argv[1]
    prescriptions_file = sys.argv[2]
    top_k = int(sys.argv[3])
    problem_type = sys.argv[4]
    model = sys.argv[5]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    story_text = Path(story_file).read_text(encoding='utf-8')

    final_story, log_file, cost_file = transform_story_single(
        story_text=story_text,
        prescriptions_file=prescriptions_file,
        top_k=top_k,
        problem_type=problem_type,
        model=model,
        temperature=0.0,
        output_dir="."
    )

    print(f"\n✓ Transformation complete!")
    print(f"  Log: {log_file}")
    print(f"  Costs: {cost_file}")
