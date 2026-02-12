"""
Abduction Module

Analyzes PyReason trace files to determine which features need improvement.
Generates ranked prescriptions for story transformation.

The abduction algorithm:
1. Compares current corpus similarity (from PyReason) with max possible (from rules)
2. Calculates confidence gaps for each feature
3. Ranks features by improvement potential
4. Creates segment-level prescriptions with duplicate removal
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required. Install with: pip install pandas")

logger = logging.getLogger(__name__)


# ==========================================================
# Score/Rating Conversion
# ==========================================================

def denormalize_score(score: float) -> int:
    """
    Convert normalized score (0.0-1.0) back to rating (1-5).

    Mapping:
        1.0 → 1 (entirely individual)
        0.75 → 2
        0.5 → 3
        0.25 → 4
        0.0 → 5 (entirely group)

    Args:
        score: Normalized score (0.0-1.0)

    Returns:
        Rating (1-5)
    """
    mapping = {
        1.0: 1,
        0.75: 2,
        0.5: 3,
        0.25: 4,
        0.0: 5
    }

    # Find closest match
    closest = min(mapping.keys(), key=lambda x: abs(x - score))
    return mapping[closest]


# ==========================================================
# Rule Loading and Parsing
# ==========================================================

def load_learned_rules(rules_file: str) -> Dict[str, Dict]:
    """
    Load and parse learned rules from file.

    Args:
        rules_file: Path to learned rules .txt file

    Returns:
        Dict mapping rule_id to rule info:
        {
            'rule_0': {
                'rule_text': '...',
                'feature': 'internal_goals',
                'confidence': 0.82,
                'threshold_min': 0.75,
                'threshold_max': 0.75
            },
            ...
        }
    """
    rules = {}

    with open(rules_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Parse rule: corpus(X, feature):[conf,conf] <-1 body
            rule_parts = line.split(' <-1 ')
            if len(rule_parts) != 2:
                logger.warning(f"Skipping malformed rule at line {i}: {line}")
                continue

            head, body = rule_parts

            # Extract feature and confidence from head
            # Format: corpus(X, feature):[conf,conf]
            head_match = re.search(r'corpus\(X,\s*(.+?)\):\[([0-9.]+),([0-9.]+)\]', head)
            if not head_match:
                logger.warning(f"Could not parse head at line {i}: {head}")
                continue

            feature = head_match.group(1).strip()
            confidence = float(head_match.group(2))

            # Extract threshold from body
            # Format: individualistic_feature(X, feature):[min,max]
            body_match = re.search(r'individualistic_feature\(X,\s*.+?\):\[([0-9.]+),([0-9.]+)\]', body)
            if not body_match:
                logger.warning(f"Could not parse body at line {i}: {body}")
                continue

            threshold_min = float(body_match.group(1))
            threshold_max = float(body_match.group(2))

            rules[f'rule_{i}'] = {
                'rule_text': line,
                'feature': feature,
                'confidence': confidence,
                'threshold_min': threshold_min,
                'threshold_max': threshold_max,
                'rule_id': i
            }

    logger.info(f"Loaded {len(rules)} learned rules")
    return rules


# ==========================================================
# Trace CSV Parsing
# ==========================================================

def parse_corpus_derivations(trace_csv_path: str) -> List[Dict]:
    """
    Parse corpus derivations from PyReason trace CSV.

    Args:
        trace_csv_path: Path to rule_trace_edges_*.csv file

    Returns:
        List of corpus derivations with feature, confidence, and rule_id
    """
    df = pd.read_csv(trace_csv_path)

    # Filter for corpus entries
    corpus_rows = df[df['Label'] == 'corpus']
    logger.info(f"Found {len(corpus_rows)} corpus derivations in trace CSV")

    corpus_derivations = []

    for _, row in corpus_rows.iterrows():
        try:
            # Parse tuple string: "('story', 'feature')"
            import ast
            edge_tuple = ast.literal_eval(row['Edge'])
            story, feature = edge_tuple

            # Parse confidence from New Bound: "[0.33, 0.33]"
            bound_match = re.search(r'\[([0-9.]+),([0-9.]+)\]', str(row['New Bound']))
            if not bound_match:
                logger.warning(f"Could not parse bounds: {row['New Bound']}")
                continue

            confidence = float(bound_match.group(1))
            rule_id = row['Occurred Due To']  # e.g., "rule_58"

            corpus_derivations.append({
                'story': story,
                'feature': feature,
                'confidence': confidence,
                'rule_id': rule_id
            })

        except Exception as e:
            logger.warning(f"Error parsing trace row: {e}")
            continue

    logger.info(f"Successfully parsed {len(corpus_derivations)} corpus derivations")
    return corpus_derivations


# ==========================================================
# Gap Analysis
# ==========================================================

def calculate_confidence_gaps(
        corpus_derivations: List[Dict],
        learned_rules: Dict[str, Dict],
        ground_atoms: Dict[str, List[float]]
) -> List[Dict]:
    """
    Calculate confidence gaps for each feature.

    Args:
        corpus_derivations: Parsed from trace CSV
        learned_rules: Loaded from rules file
        ground_atoms: Original ground atoms (to get current scores)

    Returns:
        List of gap dictionaries, sorted by confidence_gap descending
    """
    # Group rules by feature to find max confidence per feature
    rules_by_feature = defaultdict(list)
    for rule_id, rule_info in learned_rules.items():
        rules_by_feature[rule_info['feature']].append(rule_info)

    gaps = []

    # Analyze each corpus derivation
    for derivation in corpus_derivations:
        feature = derivation['feature']
        story = derivation['story']
        current_conf = derivation['confidence']

        # Find all rules for this feature
        feature_rules = rules_by_feature.get(feature, [])

        if not feature_rules:
            logger.warning(f"No rules found for feature: {feature}")
            continue

        # Find rule with highest confidence
        max_rule = max(feature_rules, key=lambda x: x['confidence'])
        gap = max_rule['confidence'] - current_conf

        # Get current score from ground atoms
        atom_key = f'individualistic_feature({story}, {feature})'
        current_score = ground_atoms.get(atom_key, [0.0, 0.0])[0]

        # Target score is the threshold from the best rule
        target_score = max_rule['threshold_min']

        # Convert to ratings
        current_rating = denormalize_score(current_score)
        target_rating = denormalize_score(target_score)

        gaps.append({
            'story': story,
            'feature': feature,
            'feature_display': feature,
            'current_confidence': current_conf,
            'max_confidence': max_rule['confidence'],
            'confidence_gap': gap,
            'current_score': current_score,
            'target_score': target_score,
            'current_rating': current_rating,
            'target_rating': target_rating,
            'current_rule': derivation['rule_id'],
            'target_rule': f"rule_{max_rule['rule_id']}"
        })

    # Sort by confidence gap (descending)
    gaps.sort(key=lambda x: x['confidence_gap'], reverse=True)

    return gaps


# ==========================================================
# Prescription Generation
# ==========================================================

def create_ranked_prescriptions(
        feature_gaps: List[Dict],
        segments_metadata: Dict[str, Dict],
        gap_threshold: float = 0.01
) -> List[Dict]:
    """
    Create ranked segment-level prescriptions with duplicate removal.

    Args:
        feature_gaps: List of feature gaps (already sorted by gap descending)
        segments_metadata: Segments metadata from graph_builder
        gap_threshold: Only include features with gap >= threshold

    Returns:
        List of prescription dictionaries with rank
    """
    prescriptions = []
    seen_segments = set()  # Track segments we've already included
    rank = 1

    for gap in feature_gaps:
        feature = gap['feature']

        # Skip if gap too small
        if gap['confidence_gap'] < gap_threshold:
            logger.debug(f"Skipping feature {feature} (gap {gap['confidence_gap']:.3f} < threshold)")
            continue

        # Get segments for this feature
        feature_meta = segments_metadata.get(feature, {})
        excerpts = feature_meta.get('excerpts', [])

        if not excerpts:
            logger.warning(
                f"Feature '{feature}' has gap {gap['confidence_gap']:.3f} "
                f"but no excerpts found in survey - skipping"
            )
            continue

        # Add each unique segment
        for segment_text in excerpts:
            # Skip if we've already added this segment
            if segment_text in seen_segments:
                logger.debug(f"Skipping duplicate segment: {segment_text[:50]}...")
                continue

            # #Skip while generating ranked prescriptsions
            # if gap['']
            # Add prescription
            prescriptions.append({
                'rank': rank,
                'feature': feature,
                'segment_text': segment_text,
                'current_score': gap['current_score'],
                'target_score': gap['target_score'],
                'current_rating': gap['current_rating'],
                'target_rating': gap['target_rating'],
                'confidence_gap': gap['confidence_gap']
            })

            seen_segments.add(segment_text)
            rank += 1

    logger.info(f"Created {len(prescriptions)} ranked prescriptions ({len(seen_segments)} unique segments)")

    return prescriptions


# ==========================================================
# Main Abduction Function
# ==========================================================

def run_abduction(
        trace_csv_path: str,
        rules_file: str,
        ground_atoms: Dict[str, List[float]],
        segments_metadata: Dict[str, Dict],
        output_dir: str,
        gap_threshold: float = 0.01
) -> Tuple[str, str, bool]:
    """
    Run abduction analysis to determine transformation priorities.

    Args:
        trace_csv_path: Path to PyReason trace CSV file
        rules_file: Path to learned rules .txt file
        ground_atoms: Ground atoms dict (for current scores)
        segments_metadata: Segments metadata from graph_builder
        output_dir: Directory to save output files
        gap_threshold: Minimum gap to consider (default: 0.01)

    Returns:
        Tuple of (abduction_json_path, prescriptions_json_path, stop_condition_met)
    """
    logger.info("=" * 60)
    logger.info("ABDUCTION ANALYSIS")
    logger.info("=" * 60)

    # Load learned rules
    logger.info("Loading learned rules...")
    learned_rules = load_learned_rules(rules_file)

    # Parse corpus derivations from trace CSV
    logger.info(f"Parsing trace CSV: {trace_csv_path}")
    corpus_derivations = parse_corpus_derivations(trace_csv_path)

    if not corpus_derivations:
        logger.error("No corpus derivations found in trace CSV")
        raise ValueError("PyReason trace contains no corpus derivations")

    # Extract story name
    stories = set(d['story'] for d in corpus_derivations)
    story_name = list(stories)[0] if stories else 'unknown'
    logger.info(f"Analyzing story: {story_name}")

    # Calculate confidence gaps
    logger.info("Calculating confidence gaps...")
    feature_gaps = calculate_confidence_gaps(
        corpus_derivations=corpus_derivations,
        learned_rules=learned_rules,
        ground_atoms=ground_atoms
    )

    # Calculate totals
    current_total = sum(d['confidence'] for d in corpus_derivations) / len(corpus_derivations)

    # Potential is max confidence per feature
    feature_max_conf = {}
    for gap in feature_gaps:
        feature_max_conf[gap['feature']] = gap['max_confidence']
    potential_total = sum(feature_max_conf.values()) / len(feature_max_conf) if feature_max_conf else 0.0

    total_improvement = potential_total - current_total

    # Check stop condition
    max_gap = max([g['confidence_gap'] for g in feature_gaps]) if feature_gaps else 0.0
    stop_condition_met = max_gap < gap_threshold

    logger.info(f"\nCurrent total similarity: {current_total:.3f}")
    logger.info(f"Potential total similarity: {potential_total:.3f}")
    logger.info(f"Total improvement potential: {total_improvement:.3f}")
    logger.info(f"Max gap: {max_gap:.3f}")
    logger.info(f"Stop condition met: {stop_condition_met} (threshold: {gap_threshold})")

    # Log top gaps
    logger.info(f"\nTop feature gaps:")
    for i, gap in enumerate(feature_gaps[:5], 1):
        logger.info(
            f"  {i}. {gap['feature']}: "
            f"current={gap['current_confidence']:.3f}, "
            f"max={gap['max_confidence']:.3f}, "
            f"gap={gap['confidence_gap']:.3f}"
        )

    # Create abduction analysis JSON
    abduction_result = {
        'story_name': story_name,
        'current_total_similarity': round(current_total, 4),
        'potential_total_similarity': round(potential_total, 4),
        'total_improvement_potential': round(total_improvement, 4),
        'feature_gaps': [
            {
                'feature': g['feature'],
                'feature_display': g['feature_display'],
                'current_confidence': round(g['current_confidence'], 4),
                'max_confidence': round(g['max_confidence'], 4),
                'confidence_gap': round(g['confidence_gap'], 4),
                'current_score': round(g['current_score'], 4),
                'target_score': round(g['target_score'], 4),
                'current_rating': g['current_rating'],
                'target_rating': g['target_rating'],
                'current_rule': g['current_rule'],
                'target_rule': g['target_rule']
            }
            for g in feature_gaps
        ],
        'stop_condition_met': stop_condition_met,
        'gap_threshold': gap_threshold
    }

    # Save abduction analysis
    abduction_file = Path(output_dir) / "abduction_analysis.json"
    with open(abduction_file, 'w', encoding='utf-8') as f:
        json.dump(abduction_result, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Abduction analysis saved to: {abduction_file}")

    # Create ranked prescriptions
    logger.info("\nCreating ranked prescriptions...")
    prescriptions = create_ranked_prescriptions(
        feature_gaps=feature_gaps,
        segments_metadata=segments_metadata,
        gap_threshold=gap_threshold
    )

    # Save ranked prescriptions
    prescriptions_result = {
        'story_name': story_name,
        'total_segments': len(prescriptions),
        'prescriptions': prescriptions
    }

    prescriptions_file = Path(output_dir) / "ranked_prescriptions.json"
    with open(prescriptions_file, 'w', encoding='utf-8') as f:
        json.dump(prescriptions_result, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Ranked prescriptions saved to: {prescriptions_file}")
    logger.info(f"  Total prescriptions: {len(prescriptions)}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ABDUCTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Features analyzed: {len(feature_gaps)}")
    logger.info(f"Segments to transform: {len(prescriptions)}")
    logger.info(
        f"Stop condition: {'YES - No more improvements needed' if stop_condition_met else 'NO - Continue transforming'}")
    logger.info("=" * 60)

    return str(abduction_file), str(prescriptions_file), stop_condition_met


# ==========================================================
# For Testing/Direct Execution
# ==========================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python abduction.py <trace.csv> <rules.txt> <ground_atoms.json> <segments_metadata.json>")
        print(
            "Example: python abduction.py traces/rule_trace_edges.csv rules.txt ground_atoms.json segments_metadata.json")
        sys.exit(1)

    trace_csv = sys.argv[1]
    rules_file = sys.argv[2]
    ground_atoms_file = sys.argv[3]
    segments_file = sys.argv[4]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load inputs
    with open(ground_atoms_file, 'r') as f:
        ground_atoms = json.load(f)

    with open(segments_file, 'r') as f:
        segments_data = json.load(f)
        segments_metadata = segments_data.get('story', {})

    # Run abduction
    abduction_file, prescriptions_file, stop = run_abduction(
        trace_csv_path=trace_csv,
        rules_file=rules_file,
        ground_atoms=ground_atoms,
        segments_metadata=segments_metadata,
        output_dir=".",
        gap_threshold=0.01
    )

    print(f"\n✓ Abduction analysis: {abduction_file}")
    print(f"✓ Ranked prescriptions: {prescriptions_file}")
    print(f"✓ Stop condition met: {stop}")