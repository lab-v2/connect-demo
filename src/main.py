"""
CONNECT Project - Main Script

This is the main entry point for the CONNECT project.
Handles both Phase 1 (training) and Phase 2 (testing/transformation).

Author: JM Patil
Date: November 11, 2025
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('.env')
# print(f"DEBUG: API key loaded: {os.getenv('OPENAI_API_KEY')[:10]}...")  # Print first 10 chars


from story_loader import load_training_stories, load_single_story
from llm_survey import conduct_surveys, conduct_survey_single_story, sanitize_model_name, CostTracker
from rule_learner import learn_rules
from graph_builder import build_ground_atoms_from_survey, save_ground_atoms, save_segments_metadata
from pyreason_runner import run_pyreason
from abduction import run_abduction
from story_transformer import transform_story_iteratively

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('connect.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CONNECT project."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='CONNECT Project - Narrative Transformation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Train on individualistic stories (forward problem)
  python main.py --phase 1 --problem forward --data-dir data --output-dir output/phase1

  # Phase 1: Train on collectivistic stories (inverse problem)
  python main.py --phase 1 --problem inverse --data-dir data --output-dir output/phase1

  # Phase 2: Transform collectivistic story to individualistic
  python main.py --phase 2 --problem forward --rules output/phase1/rules.txt --story data/test_story.txt

  # Phase 2: Transform individualistic story to collectivistic
  python main.py --phase 2 --problem inverse --rules output/phase1/rules.txt --story data/test_story.txt
        """
    )

    # Core arguments
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        required=True,
        help='Phase 1 (training) or Phase 2 (testing/transformation)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='forward: individualistic->collectivistic, inverse: collectivistic->individualistic'
    )

    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Path to data directory (default: data)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Path to output directory (default: output)'
    )

    # Phase 1 specific arguments
    parser.add_argument(
        '--skip-survey',
        action='store_true',
        help='Skip LLM survey step and use existing results (works for both Phase 1 and Phase 2)'
    )

    parser.add_argument(
        '--survey-results',
        type=str,
        help='[Phase 1] Path to existing survey results JSON file'
    )

    # Phase 2 specific arguments
    parser.add_argument(
        '--story',
        type=str,
        help='[Phase 2] Path to test story file to transform'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='[Phase 2] Maximum transformation iterations (default: 3)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='[Phase 2] Number of top features to transform per iteration (default: 3)'
    )

    parser.add_argument(
        '--rules',
        type=str,
        default=None,
        help='[Phase 2] Path to learned rules file (optional - auto-determined if not provided)'
    )

    # LLM configuration
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='LLM model to use (default: gpt-4o)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature (default: 0.0)'
    )

    # General options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)

    # Log configuration
    logger.info("=" * 60)
    logger.info(f"CONNECT Project - Phase {args.phase}")
    logger.info(f"Problem: {args.problem}")
    logger.info("=" * 60)

    # Execute appropriate phase
    try:
        if args.phase == 1:
            run_phase1(args)
        else:  # args.phase == 2
            run_phase2(args)

        logger.info("=" * 60)
        logger.info("Execution completed successfully!")
        logger.info("=" * 60)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        logger.error("Check connect.log for details")
        sys.exit(1)


def validate_arguments(args):
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        bool: True if valid, False otherwise
    """
    # Phase 1 validation
    if args.phase == 1:
        # Only check for data_dir if not skipping survey
        if not args.skip_survey:
            if not Path(args.data_dir).exists():
                logger.error(f"Data directory not found: {args.data_dir}")
                return False

        # No need to validate survey_results for Phase 1
        # (we load from output_dir structure instead)

    # Phase 2 validation
    if args.phase == 2:
        if not args.story:
            logger.error("--story is required for Phase 2")
            return False

        if not Path(args.story).exists():
            logger.error(f"Story file not found: {args.story}")
            return False

        # Rules path will be auto-determined if not provided
        # No need to validate here
    return True


def run_phase1(args):
    """
    Execute Phase 1: Training - Learn rules from training stories.

    This phase:
    1. Loads training stories (individualistic or collectivistic based on problem)
    2. Conducts LLM surveys on each story (20 features)
    3. Learns PyReason rules from survey results

    Args:
        args: Command line arguments
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: TRAINING")
    logger.info("=" * 60)


    if not args.skip_survey:
        #   : Import and call story_loader.py
        stories = load_training_stories(args.data_dir, args.problem)
        logger.info(f"Loaded {len(stories)} training stories")

        #   : Import and call llm_survey.py (unless skipping)
        # Determine questions file path based on problem type
        if args.problem == "forward":
            questions_file = Path(args.data_dir) / "individualistic_questions.json"
        else:  # inverse
            questions_file = Path(args.data_dir) / "collectivistic_questions.json"

        if not questions_file.exists():
            logger.error(f"Questions file not found: {questions_file}")
            sys.exit(1)


        survey_results, failed_stories = conduct_surveys(
            stories=stories,
            questions_file=str(questions_file),
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            output_dir=args.output_dir
        )

        if failed_stories:
            logger.warning(f"{len(failed_stories)} stories failed. Check failed_stories.json")
    else:
        # Load existing survey results
        logger.info("Skipping survey step (--skip-survey flag set)")
        # with open(args.survey_results, 'r') as f:
        #     survey_results = json.load(f)


    #   : Import and call rule_learner.py
    # Learn PyReason rules from survey results

    sanitized_model = sanitize_model_name(args.model)
    survey_results_dir = Path(args.output_dir) / sanitized_model / args.problem / "survey_results"

    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Learning PyReason Rules")
    logger.info("=" * 60)

    rules_file = learn_rules(
        survey_results_dir=str(survey_results_dir),
        problem_type=args.problem,
        model=args.model,
        output_dir=args.output_dir
    )

    logger.info(f"\n✓ Rules learned and saved to: {rules_file}")
    logger.info("\nPhase 1 completed successfully!")
    logger.info(f"Output directory: {args.output_dir}")


def run_phase2(args):
    """
    Execute Phase 2: Testing/Transformation - Transform test story.

    This phase:
    1. Loads test story
    2. Iteratively:
       - Surveys story
       - Builds ground atoms
       - Runs PyReason
       - Runs abduction
       - Transforms story
    3. Selects best iteration

    Args:
        args: Command line arguments
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: TESTING/TRANSFORMATION")
    logger.info("=" * 60)

    # Step 1: Load test story
    logger.info("\n[Step 1/7] Loading test story...")
    story = load_single_story(args.story)
    logger.info(f"✓ Loaded test story: {story['name']}")

    # Step 2: Load learned rules from Phase 1
    logger.info("\n[Step 2/7] Loading learned rules...")

    # Auto-determine rules path if not provided
    if args.rules:
        rules_file = Path(args.rules)
    else:
        # Default: output/phase1/{model}/{problem}/learned_rules/pyreason_rules.txt
        sanitized_model = sanitize_model_name(args.model)
        rules_file = Path("output/phase1") / sanitized_model / args.problem / "learned_rules" / "pyreason_rules.txt"
        logger.info(f"Auto-determined rules path: {rules_file}")

    if not rules_file.exists():
        logger.error(f"Rules file not found: {rules_file}")
        logger.error("Please ensure Phase 1 has been run for this model and problem type,")
        logger.error("or provide explicit --rules path")
        sys.exit(1)

    logger.info(f"✓ Rules loaded from: {rules_file}")

    # Create output directory structure
    sanitized_model = sanitize_model_name(args.model)
    story_output_dir = Path(args.output_dir) / "phase2" / sanitized_model / args.problem / story['name']
    story_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Output directory: {story_output_dir}")

    # Initialize cost tracker for this story
    cost_tracker = CostTracker(args.model)
    logger.info("✓ Cost tracker initialized")


    # Determine questions file based on problem type
    # Forward problem: always use individualistic questions (whether training or testing)
    # Inverse problem: always use collectivistic questions (whether training or testing)
    if args.problem == "forward":
        questions_file = Path(args.data_dir) / "individualistic_questions.json"
    else:  # inverse
        questions_file = Path(args.data_dir) / "collectivistic_questions.json"

    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        sys.exit(1)

    # Initialize: current story starts as the original test story
    current_story_text = story['content']
    current_story_path = story['path']

    # Main iteration loop
    for iteration in range(args.max_iterations):
        logger.info("\n" + "=" * 60)
        logger.info(f"ITERATION {iteration + 1}/{args.max_iterations}")
        logger.info("=" * 60)

        # Create iteration directory
        iter_dir = story_output_dir / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Save current story text
        story_file = iter_dir / "story.txt"
        story_file.write_text(current_story_text, encoding='utf-8')
        logger.info(f"✓ Saved current story to: {story_file}")

        # Step 3: Survey current story (or load existing)
        survey_file = iter_dir / "survey.json"

        if args.skip_survey and survey_file.exists():
            logger.info(f"\n[Step 3/7] Loading existing survey (--skip-survey)...")
            with open(survey_file, 'r', encoding='utf-8') as f:
                survey_result = json.load(f)
            logger.info(f"✓ Survey loaded from: {survey_file}")
        else:
            logger.info(f"\n[Step 3/7] Surveying story (iteration {iteration})...")

            # Create temporary story dict for survey
            temp_story = {
                'name': story['name'],
                'path': str(story_file),
                'content': current_story_text
            }

            survey_result = conduct_survey_single_story(
                story=temp_story,
                questions_file=str(questions_file),
                problem_type=args.problem,
                model=args.model,
                temperature=args.temperature,
                cost_tracker=cost_tracker
            )

            # Save survey result
            with open(survey_file, 'w', encoding='utf-8') as f:
                json.dump(survey_result, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Survey completed and saved to: {survey_file}")

        # Save cost tracking for this iteration
        iter_cost_file = iter_dir / "cost_tracking.json"
        cost_tracker.save(iter_cost_file)
        logger.info(f"✓ Cost tracking saved to: {iter_cost_file}")

        #   : Step 4: Build ground atoms from survey
        # Step 4: Build ground atoms from survey
        logger.info(f"\n[Step 4/7] Building ground atoms from survey...")

        ground_atoms, segments_metadata = build_ground_atoms_from_survey(
            survey_json_path=str(survey_file),
            story_name=story['name']
        )

        # Save ground atoms
        ground_atoms_file = iter_dir / "ground_atoms.json"
        save_ground_atoms(ground_atoms, str(ground_atoms_file))
        logger.info(f"✓ Ground atoms saved: {len(ground_atoms)} atoms")

        # Save segments metadata
        segments_metadata_file = iter_dir / "segments_metadata.json"
        save_segments_metadata(segments_metadata, str(segments_metadata_file))
        logger.info(f"✓ Segments metadata saved: {len(segments_metadata)} features")
        #   : Step 5: Run PyReason
        # Step 5: Run PyReason with ground atoms and rules
        logger.info(f"\n[Step 5/7] Running PyReason...")

        trace_dir = run_pyreason(
            ground_atoms=ground_atoms,
            rules_file=str(rules_file),
            story_name=story['name'],
            output_dir=str(iter_dir)
        )

        logger.info(f"✓ PyReason traces saved to: {trace_dir}")
        #   : Step 6: Run abduction
        # Step 6: Run abduction to prioritize features
        logger.info(f"\n[Step 6/7] Running abduction analysis...")

        # Find the trace CSV file
        trace_files = sorted(Path(trace_dir).glob("rule_trace_edges_*.csv"))
        if not trace_files:
            logger.error("No trace CSV files found from PyReason")
            sys.exit(1)

        trace_csv_path = str(trace_files[0])  # Use most recent
        logger.debug(f"Using trace file: {trace_files[0].name}")

        abduction_file, prescriptions_file, stop_condition_met = run_abduction(
            trace_csv_path=trace_csv_path,
            rules_file=str(rules_file),
            ground_atoms=ground_atoms,
            segments_metadata=segments_metadata,
            output_dir=str(iter_dir),
            gap_threshold=0.01
        )

        logger.info(f"✓ Abduction analysis saved to: {abduction_file}")
        logger.info(f"✓ Ranked prescriptions saved to: {prescriptions_file}")

        # Check stop condition
        if stop_condition_met:
            logger.info("\n🛑 Stop condition met - all gaps < 0.01")
            logger.info("No further transformation needed!")
            break  # Exit iteration loop
        #   : Step 7: Transform story
        # Step 7: Transform story using top-k features
        logger.info(f"\n[Step 7/7] Transforming story (top-{args.top_k} features)...")

        transformed_story, log_file, cost_file = transform_story_iteratively(
            story_text=current_story_text,
            prescriptions_file=prescriptions_file,
            top_k=args.top_k,
            problem_type=args.problem,
            model=args.model,
            temperature=args.temperature,
            output_dir=str(iter_dir)
        )

        logger.info(f"✓ Story transformed")
        logger.info(f"✓ Transformation log: {log_file}")
        logger.info(f"✓ Transformation costs: {cost_file}")

        # Update current story for next iteration
        current_story_text = transformed_story

        # # For now, break after first iteration (we'll add more steps later)
        # logger.info("\n⚠️  Only survey step implemented so far. Breaking after iteration 0.")
        # break
    # After the for loop ends, before "Phase 2 completed!"
    # Combine survey and transformation costs
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 FINAL COST SUMMARY")
    logger.info("=" * 60)

    # This will show combined costs from survey + transformation
    # (Both cost trackers save separately, we just log summary here)

    # Save final cost summary for entire story transformation
    final_cost_file = story_output_dir / "cost_tracking_final.json"
    cost_tracker.save(final_cost_file)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 COST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total stories processed: 1")
    logger.info(f"Total iterations: {iteration + 1}")
    logger.info(f"Total cost: ${cost_tracker.total_cost:.4f}")
    logger.info(f"Total tokens: {cost_tracker.total_input_tokens + cost_tracker.total_output_tokens:,}")
    logger.info(f"  - Input: {cost_tracker.total_input_tokens:,}")
    logger.info(f"  - Output: {cost_tracker.total_output_tokens:,}")
    logger.info(f"Cost tracking saved to: {final_cost_file}")
    logger.info("=" * 60)

    logger.info("\nPhase 2 completed!")
    logger.info(f"Output directory: {story_output_dir}")



if __name__ == "__main__":
    main()
