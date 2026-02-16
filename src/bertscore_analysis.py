#!/usr/bin/env python3
"""
BERTScore analysis script for comparing abduction vs baseline story transformations.
Compares transformed stories against the original story to measure semantic preservation.
"""

import os
import csv
import argparse
from pathlib import Path
from bert_score import score
import pandas as pd


def read_story_file(filepath):
    """Read a story file and return its content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def compute_bertscore(original_text, candidate_text):
    """
    Compute BERTScore between original and candidate texts.
    Returns precision, recall, f1 scores.
    """
    if not original_text or not candidate_text:
        return None, None, None

    try:
        P, R, F1 = score(
            [candidate_text],
            [original_text],
            lang="en",
            verbose=False
        )
        return float(P[0]), float(R[0]), float(F1[0])
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return None, None, None


def process_stories(data_root, original_direction, model_name, output_csv):
    """
    Process all stories in the evaluation_data directory.

    Args:
        data_root: Path to evaluation_data root directory
        original_direction: "collectivistic" or "individualistic"
        model_name: Name of the model folder (e.g., "claude-sonnet-4-5")
        output_csv: Path to output CSV file
    """

    # Determine the transformation direction
    if original_direction == "collectivistic":
        transform_direction = "individualistic"
    elif original_direction == "individualistic":
        transform_direction = "collectivistic"
    else:
        raise ValueError("original_direction must be 'collectivistic' or 'individualistic'")

    # Build paths
    data_root = Path(data_root)
    model_folder = data_root / model_name

    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    original_dir = model_folder / original_direction
    transform_dir = model_folder / transform_direction

    if not original_dir.exists():
        raise FileNotFoundError(f"Original direction folder not found: {original_dir}")
    if not transform_dir.exists():
        raise FileNotFoundError(f"Transform direction folder not found: {transform_dir}")

    # Prepare output
    os.makedirs(Path(output_csv).parent, exist_ok=True)

    results = []

    # Iterate through story folders in the original direction
    for story_folder in sorted(original_dir.iterdir()):
        if not story_folder.is_dir():
            continue

        story_name = story_folder.name

        # Read original story
        original_story_path = story_folder / "original_story.txt"
        original_text = read_story_file(original_story_path)

        if not original_text:
            print(f"Skipping {story_name}: Could not read original story")
            continue

        # Read abduction transformed story
        abduction_path = story_folder / "abduction_transformed.txt"
        abduction_text = read_story_file(abduction_path)

        # Read baseline transformed story
        baseline_path = story_folder / "baseline_transformed.txt"
        baseline_text = read_story_file(baseline_path)

        if not abduction_text and not baseline_text:
            print(f"Skipping {story_name}: Could not read any transformed stories")
            continue

        # Compute BERTScores
        print(f"Processing: {story_name}")

        ab_precision, ab_recall, ab_f1 = None, None, None
        bs_precision, bs_recall, bs_f1 = None, None, None

        if abduction_text:
            ab_precision, ab_recall, ab_f1 = compute_bertscore(original_text, abduction_text)
            if ab_f1 is not None:
                print(f"  Abduction - P: {ab_precision:.4f}, R: {ab_recall:.4f}, F1: {ab_f1:.4f}")

        if baseline_text:
            bs_precision, bs_recall, bs_f1 = compute_bertscore(original_text, baseline_text)
            if bs_f1 is not None:
                print(f"  Baseline  - P: {bs_precision:.4f}, R: {bs_recall:.4f}, F1: {bs_f1:.4f}")

        # Store results
        results.append({
            'story_name': story_name,
            'original_direction': original_direction,
            'abduction_bertscore_precision': ab_precision if ab_precision is not None else '',
            'abduction_bertscore_recall': ab_recall if ab_recall is not None else '',
            'abduction_bertscore_f1': ab_f1 if ab_f1 is not None else '',
            'baseline_bertscore_precision': bs_precision if bs_precision is not None else '',
            'baseline_bertscore_recall': bs_recall if bs_recall is not None else '',
            'baseline_bertscore_f1': bs_f1 if bs_f1 is not None else ''
        })

    # Write CSV
    if results:
        fieldnames = [
            'story_name',
            'original_direction',
            'abduction_bertscore_precision',
            'abduction_bertscore_recall',
            'abduction_bertscore_f1',
            'baseline_bertscore_precision',
            'baseline_bertscore_recall',
            'baseline_bertscore_f1'
        ]

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Results saved to: {output_csv}")
        print(f"✓ Processed {len(results)} stories")

        # Print summary statistics
        df = pd.read_csv(output_csv)
        print("\n=== Summary Statistics ===")
        print(
            f"Abduction F1 - Mean: {df['abduction_bertscore_f1'].mean():.4f}, Std: {df['abduction_bertscore_f1'].std():.4f}")
        print(
            f"Baseline F1  - Mean: {df['baseline_bertscore_f1'].mean():.4f}, Std: {df['baseline_bertscore_f1'].std():.4f}")
    else:
        print("No results to save.")


def main():
    parser = argparse.ArgumentParser(
        description="Compute BERTScores for story transformations"
    )
    parser.add_argument(
        "original_direction",
        choices=["collectivistic", "individualistic"],
        help="Direction of original stories"
    )
    parser.add_argument(
        "model_name",
        help="Name of the model folder (e.g., claude-sonnet-4-5, bedrock-us.deepseek.r1-v1-0)"
    )
    parser.add_argument(
        "--data-root",
        default="evaluation_data",
        help="Path to evaluation_data root directory (default: evaluation_data)"
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to output CSV file (if not provided, auto-generated based on model and direction)"
    )

    args = parser.parse_args()

    # Generate filename if not provided
    if args.output_csv is None:
        # Determine problem direction
        if args.original_direction == "collectivistic":
            direction = "forward"
        else:
            direction = "inverse"

        filename = f"bertscore_results_{args.model_name}_{direction}.csv"
        args.output_csv = f"output_analysis/bert_scores_csv/{filename}"

    print(f"Starting BERTScore analysis...")
    print(f"Original direction: {args.original_direction}")
    print(f"Model: {args.model_name}")
    print(f"Data root: {args.data_root}")
    print(f"Output CSV: {args.output_csv}\n")

    process_stories(args.data_root, args.original_direction, args.model_name, args.output_csv)


if __name__ == "__main__":
    main()