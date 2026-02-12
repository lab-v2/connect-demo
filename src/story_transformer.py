"""
Story Transformer Module

Transforms stories by rewriting segments identified by abduction analysis.
Uses LLM to iteratively modify segments to shift narrative style.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from llm_survey import call_llm_with_retry, CostTracker

logger = logging.getLogger(__name__)

# ==========================================================
# Prompt Templates (from their utils.py)
# ==========================================================

COLLECTIVIST = "collectivist"
INDIVIDUALIST = "individualist"

PROMPT_TEMPLATE = '''You are given a {source_type} story below. 



{story}



Your goal is to make the story more {target_type}. To make more {target_type}, you will update only the selected segment from the story which is provided below. 

Selected segment: ** {segment} **

Now, rewrite the whole story by updating only the selected segment to make the story more {target_type}. Don't change other part of the story, and just output the rewritten story, nothing else.
'''


# ==========================================================
# Prescription Processing
# ==========================================================

def group_prescriptions_by_feature(prescriptions: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group prescriptions by feature while maintaining rank order.

    Args:
        prescriptions: List of prescription dicts (already ranked)

    Returns:
        Dict mapping feature name to list of its prescriptions (in rank order)
    """
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
    """
    Select top-k features and their prescriptions.

    Args:
        prescriptions: List of prescription dicts (ranked by gap)
        top_k: Number of top features to select

    Returns:
        Tuple of (list of selected feature names, list of selected prescriptions)
    """
    # Group by feature
    feature_groups = group_prescriptions_by_feature(prescriptions)

    # Get features in order of first appearance (which is by gap since sorted)
    seen_features = []
    for presc in prescriptions:
        if presc['feature'] not in seen_features:

            if problem_type == "forward":
                # In forward problem, only consider features where target < current
                if presc['target_rating'] < presc['current_rating']:
                    seen_features.append(presc['feature'])
            elif problem_type == "inverse":
                # In inverse problem, only consider features where target > current
                if presc['target_rating'] > presc['current_rating']:
                    seen_features.append(presc['feature'])
            # seen_features.append(presc['feature'])

    # Take top-k features
    selected_features = seen_features[:top_k]

    # Get all prescriptions for these features
    selected_prescriptions = []
    for feature in selected_features:
        selected_prescriptions.extend(feature_groups[feature])

    # Re-sort by original rank
    selected_prescriptions.sort(key=lambda x: x['rank'])

    logger.info(f"Selected top {len(selected_features)} features:")
    for i, feature in enumerate(selected_features, 1):
        num_segments = len(feature_groups[feature])
        logger.info(f"  {i}. {feature}: {num_segments} segments")

    logger.info(f"Total segments to transform: {len(selected_prescriptions)}")

    return selected_features, selected_prescriptions


# ==========================================================
# Prompt Creation
# ==========================================================

def create_transformation_prompt(
        story_text: str,
        segment_text: str,
        problem_type: str
) -> str:
    """
    Create transformation prompt for a segment.

    Args:
        story_text: Current full story text
        segment_text: Segment to transform
        problem_type: "forward" or "inverse"

    Returns:
        Formatted prompt string
    """
    # Determine source and target types
    if problem_type == "forward":
        source_type = COLLECTIVIST
        target_type = INDIVIDUALIST
    else:  # inverse
        source_type = INDIVIDUALIST
        target_type = COLLECTIVIST

    # Format prompt
    prompt = PROMPT_TEMPLATE.format(
        source_type=source_type,
        target_type=target_type,
        story=story_text,
        segment=segment_text
    )

    return prompt


# ==========================================================
# Story Transformation
# ==========================================================

def transform_story_iteratively(
        story_text: str,
        prescriptions_file: str,
        top_k: int,
        problem_type: str,
        model: str,
        temperature: float,
        output_dir: str
) -> Tuple[str, str, str]:
    """
    Transform story by iteratively rewriting segments.

    Args:
        story_text: Original story text
        prescriptions_file: Path to ranked_prescriptions.json
        top_k: Number of top features to use
        problem_type: "forward" or "inverse"
        model: LLM model name
        temperature: LLM temperature
        output_dir: Directory to save outputs

    Returns:
        Tuple of (final_story, transformation_log_path, cost_tracking_path)
    """
    logger.info("=" * 60)
    logger.info("STORY TRANSFORMATION")
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

    # Initialize cost tracker for transformations
    cost_tracker = CostTracker(model)

    # Track transformations
    transformation_log = {
        'story_name': story_name,
        'problem_type': problem_type,
        'model': model,
        'temperature': temperature,
        'k_features': len(selected_features),
        'n_segments': len(selected_prescriptions),
        'transformations': []
    }

    # Transform story iteratively
    current_story = story_text
    # segments_transformed = 0  # Track actual transformations (excluding skipped)

    for i, presc in enumerate(selected_prescriptions, 1):
        logger.info(f"\n[Transformation {i}/{len(selected_prescriptions)}]")
        logger.info(f"  Feature: {presc['feature']}")
        logger.info(f"  Segment: {presc['segment_text'][:60]}...")
        logger.info(f"  Target: {presc['current_rating']} → {presc['target_rating']}")

        # Check for reversal (prevent rating from going wrong direction)
        current_rating = presc['current_rating']
        target_rating = presc['target_rating']

        is_reversal = False
        if problem_type == "forward":
            # Forward: ratings should only decrease (become more individual: 5→1)
            if target_rating > current_rating:
                is_reversal = True
                logger.warning(
                    f"  ⚠️  SKIPPING REVERSAL: {presc['feature']} "
                    f"rating {current_rating} → {target_rating} not allowed in forward problem"
                )
        else:  # inverse
            # Inverse: ratings should only increase (become more collectivistic: 1→5)
            if target_rating < current_rating:
                is_reversal = True
                logger.warning(
                    f"  ⚠️  SKIPPING REVERSAL: {presc['feature']} "
                    f"rating {current_rating} → {target_rating} not allowed in inverse problem"
                )

        # Skip this segment if it's a reversal
        if is_reversal:
            step_log = {
                'step': i,
                'feature': presc['feature'],
                'segment': presc['segment_text'],
                'current_rating': current_rating,
                'target_rating': target_rating,
                'status': 'skipped_reversal',
                'reason': f"Rating {current_rating}→{target_rating} not allowed in {problem_type} problem"
            }
            transformation_log['transformations'].append(step_log)
            continue  # Skip to next prescription


        # Create prompt
        prompt = create_transformation_prompt(
            story_text=current_story,
            segment_text=presc['segment_text'],
            problem_type=problem_type
        )

        try:
            # Call LLM
            rewritten_story, input_tokens, output_tokens = call_llm_with_retry(
                system_prompt="You are a literary expert.",
                user_prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=4096,
                max_retries=3
            )

            # Track cost
            cost_tracker.add_call(
                story_name=f"{story_name}_step_{i}",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            # Log this transformation
            step_log = {
                'step': i,
                'feature': presc['feature'],
                'segment': presc['segment_text'],
                'current_rating': presc['current_rating'],
                'target_rating': presc['target_rating'],
                'confidence_gap': presc['confidence_gap'],
                'prompt_used': prompt,
                'tokens': {
                    'input': input_tokens,
                    'output': output_tokens
                },
                'cost': round(cost_tracker.per_story_breakdown[-1]['cost_usd'], 6)
            }
            transformation_log['transformations'].append(step_log)

            # Update current story for next iteration
            current_story = rewritten_story
            # segments_transformed += 1  # ← ADD THIS LINE

            logger.info(f"  ✓ Transformed (tokens: {input_tokens + output_tokens})")

        except Exception as e:
            logger.error(f"  ✗ Transformation failed: {e}")
            # Log the failure
            step_log = {
                'step': i,
                'feature': presc['feature'],
                'segment': presc['segment_text'],
                'error': str(e)
            }
            transformation_log['transformations'].append(step_log)
            # Continue with other segments
            continue

    # Save final transformed story
    story_file = Path(output_dir) / "story_transformed.txt"
    story_file.write_text(current_story, encoding='utf-8')
    logger.info(f"\n✓ Final transformed story saved to: {story_file}")

    # Save transformation log
    log_file = Path(output_dir) / "transformation_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(transformation_log, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Transformation log saved to: {log_file}")

    # Save cost tracking with k_features and n_segments
    cost_data = cost_tracker.to_dict()
    cost_data['k_features'] = len(selected_features)
    cost_data['n_segments'] = len(selected_prescriptions)
    cost_data['successful_transformations'] = sum(
        1 for t in transformation_log['transformations'] if 'error' not in t
    )

    cost_file = Path(output_dir) / "cost_tracking_transformation.json"
    with open(cost_file, 'w', encoding='utf-8') as f:
        json.dump(cost_data, f, indent=2)
    logger.info(f"✓ Transformation costs saved to: {cost_file}")

    # Summary
    successful_transforms = sum(
        1 for t in transformation_log['transformations']
        if t.get('status') != 'skipped_reversal' and 'error' not in t
    )
    skipped_reversals = sum(
        1 for t in transformation_log['transformations']
        if t.get('status') == 'skipped_reversal'
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRANSFORMATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Features selected: {len(selected_features)}")
    logger.info(f"Segments attempted: {len(selected_prescriptions)}")
    logger.info(f"Segments transformed: {successful_transforms}")
    logger.info(f"Segments skipped (reversals): {skipped_reversals}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info("=" * 60)

    return current_story, str(log_file), str(cost_file)


# ==========================================================
# For Testing/Direct Execution
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python story_transformer.py <story.txt> <prescriptions.json> <top_k> <problem_type> <model>")
        print("Example: python story_transformer.py story.txt ranked_prescriptions.json 3 forward gpt-4o")
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

    # Load story
    story_text = Path(story_file).read_text(encoding='utf-8')

    # Transform
    final_story, log_file, cost_file = transform_story_iteratively(
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