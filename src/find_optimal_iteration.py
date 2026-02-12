"""
Find Optimal Iteration Script

For each story, finds which iteration achieved the best rating
(minimum for forward problem, maximum for inverse problem).

Creates a summary CSV with optimal ratings and all iteration values.

Output: CSV saved to output_analysis/phase2/{model}/{problem}/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================
# Optimal Iteration Finding
# ==========================================================

def find_optimal_iterations(
    model: str,
    problem_type: str,
    max_iterations: int,
    phase2_analysis_dir: str = "output_analysis/phase2"
) -> pd.DataFrame:
    """
    Find optimal iteration for each story.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        max_iterations: Maximum iterations used
        phase2_analysis_dir: Directory with extracted results

    Returns:
        DataFrame with optimal iterations
    """
    logger.info("=" * 60)
    logger.info("FINDING OPTIMAL ITERATIONS")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")
    logger.info(f"Max iterations: {max_iterations}")

    # Sanitize model name
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Load original ratings
    analysis_dir = Path(phase2_analysis_dir) / sanitized_model / problem_type
    original_file = analysis_dir / "original_ratings.csv"

    if not original_file.exists():
        raise FileNotFoundError(f"Original ratings not found: {original_file}")

    df_original = pd.read_csv(original_file)
    logger.info(f"Loaded original ratings: {len(df_original)} stories")

    # Load all iteration files
    iteration_dfs = []
    for i in range(max_iterations - 1):  # 0 to N-2
        iter_file = analysis_dir / f"transformed_ratings_iteration_{i}.csv"
        if iter_file.exists():
            df_iter = pd.read_csv(iter_file)
            iteration_dfs.append((i, df_iter))
            logger.info(f"Loaded iteration {i}: {len(df_iter)} stories")
        else:
            logger.warning(f"Iteration {i} file not found: {iter_file}")

    if not iteration_dfs:
        raise ValueError("No iteration files found")

    # Process each story
    results = []

    for _, row in df_original.iterrows():
        story_name = row['story_name']
        original_mean = row['Mean']
        original_median = row['Median']
        original_mode = row['Mode']

        # Collect Mean ratings from all iterations to find optimal
        iter_means = {}
        for iter_num, df_iter in iteration_dfs:
            story_row = df_iter[df_iter['story_name'] == story_name]
            if len(story_row) > 0:
                iter_means[iter_num] = story_row['Mean'].values[0]
            else:
                iter_means[iter_num] = np.nan

        # Find optimal iteration based on Mean
        if problem_type == "forward":
            # Forward: minimum Mean is best (closer to 1 = more individualistic)
            valid_means = {k: v for k, v in iter_means.items() if not pd.isna(v)}
            if valid_means:
                optimal_iter = min(valid_means, key=valid_means.get)
            else:
                optimal_iter = np.nan
        else:  # inverse
            # Inverse: maximum Mean is best (closer to 5 = more collectivistic)
            valid_means = {k: v for k, v in iter_means.items() if not pd.isna(v)}
            if valid_means:
                optimal_iter = max(valid_means, key=valid_means.get)
            else:
                optimal_iter = np.nan

        # Extract all three statistics from the optimal iteration
        if not pd.isna(optimal_iter):
            optimal_df = iteration_dfs[optimal_iter][1]  # Get the DataFrame
            optimal_story = optimal_df[optimal_df['story_name'] == story_name]

            if len(optimal_story) > 0:
                optimal_mean = optimal_story['Mean'].values[0]
                optimal_median = optimal_story['Median'].values[0]
                optimal_mode = optimal_story['Mode'].values[0]
            else:
                optimal_mean = np.nan
                optimal_median = np.nan
                optimal_mode = np.nan
        else:
            optimal_mean = np.nan
            optimal_median = np.nan
            optimal_mode = np.nan

        # Build result row
        result = {
            'story_name': story_name,
            'optimal_mean': round(optimal_mean, 2) if not pd.isna(optimal_mean) else np.nan,
            'optimal_median': round(optimal_median, 2) if not pd.isna(optimal_median) else np.nan,
            'optimal_mode': round(optimal_mode, 2) if not pd.isna(optimal_mode) else np.nan,
            'optimal_iteration': int(optimal_iter) if not pd.isna(optimal_iter) else np.nan,
            'original_mean': round(original_mean, 2),
            'original_median': round(original_median, 2),
            'original_mode': round(original_mode, 2)
        }

        # Add all iteration means for reference
        for iter_num, df_iter in iteration_dfs:
            col_name = f'abductive_iter_{iter_num}_mean'
            result[col_name] = round(iter_means.get(iter_num, np.nan), 2) if not pd.isna(iter_means.get(iter_num, np.nan)) else np.nan

        results.append(result)

    # Create DataFrame
    df_optimal = pd.DataFrame(results)

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMAL ITERATIONS SUMMARY")
    logger.info("=" * 60)

    if not df_optimal['optimal_iteration'].isna().all():
        iteration_counts = df_optimal['optimal_iteration'].value_counts().sort_index()
        logger.info("Stories achieving optimal at each iteration:")
        for iter_num, count in iteration_counts.items():
            logger.info(f"  Iteration {int(iter_num)}: {count} stories")

        avg_improvement_mean = (df_optimal['original_mean'] - df_optimal['optimal_mean']).mean()
        logger.info(f"\nAverage improvement (Mean): {avg_improvement_mean:.2f} rating points")

    return df_optimal


# ==========================================================
# Main Function
# ==========================================================

def create_optimal_iteration_csv(
    model: str,
    problem_type: str,
    max_iterations: int,
    phase2_analysis_dir: str = "output_analysis/phase2"
):
    """
    Create optimal iteration CSV.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        max_iterations: Maximum iterations
        phase2_analysis_dir: Analysis directory
    """
    # Find optimal iterations
    df_optimal = find_optimal_iterations(
        model=model,
        problem_type=problem_type,
        max_iterations=max_iterations,
        phase2_analysis_dir=phase2_analysis_dir
    )

    # Save to CSV
    sanitized_model = model.replace("/", "-").replace(":", "-")
    output_dir = Path(phase2_analysis_dir) / sanitized_model / problem_type
    output_file = output_dir / "optimal_iterations.csv"

    df_optimal.to_csv(output_file, index=False)
    logger.info(f"\n✓ Optimal iterations CSV saved to: {output_file}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Find optimal iteration for each story',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find optimal for gpt-4o forward
  python plot_scripts/find_optimal_iteration.py --model gpt-4o --problem forward --max-iterations 3
  
  # Find for Claude inverse
  python plot_scripts/find_optimal_iteration.py --model claude-sonnet-4-5 --problem inverse --max-iterations 3
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Problem type'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        required=True,
        help='Maximum iterations used in Phase 2'
    )

    parser.add_argument(
        '--phase2-analysis-dir',
        type=str,
        default='output_analysis/phase2',
        help='Phase 2 analysis directory (default: output_analysis/phase2)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create CSV
    try:
        create_optimal_iteration_csv(
            model=args.model,
            problem_type=args.problem,
            max_iterations=args.max_iterations,
            phase2_analysis_dir=args.phase2_analysis_dir
        )

        logger.info("\n✓ Processing complete!")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()