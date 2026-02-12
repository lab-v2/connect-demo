"""
Story Loader Module

Loads training stories from the data directory based on problem type.
- Forward problem: Loads individualistic stories
- Inverse problem: Loads collectivistic stories
"""

import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_training_stories(data_dir: str, problem_type: str) -> List[Dict[str, str]]:
    """
    Load training stories based on problem type.

    Args:
        data_dir: Path to data directory (e.g., "data")
        problem_type: Either "forward" or "inverse"
            - forward: Load from data/individualistic-rags-to-riches-stories/
            - inverse: Load from data/collectivistic-stories-all/

    Returns:
        List of dictionaries, each containing:
            {
                'name': str,      # Story name (filename without .txt)
                'path': str,      # Full path to the story file
                'content': str    # Story text content
            }

    Raises:
        FileNotFoundError: If story directory doesn't exist
        ValueError: If no stories found in directory
    """
    # Determine which directory to load from
    if problem_type == "forward":
        # story_subdir = "individualistic-rags-to-riches-stories"
        story_subdir = "individualistic-rags-to-riches-stories-subset-subset"
        logger.info("Loading individualistic stories for forward problem")
    else:  # inverse
        # story_subdir = "collectivistic-stories-subset"
        story_subdir = "collectivistic-stories-all"
        logger.info("Loading collectivistic stories for inverse problem")

    # Construct full path to story directory
    story_dir = Path(data_dir) / story_subdir

    # Check if directory exists
    if not story_dir.exists():
        raise FileNotFoundError(f"Story directory not found: {story_dir}")

    # Find all .txt files in the directory
    story_files = sorted(story_dir.glob("*.txt"))

    if len(story_files) == 0:
        raise ValueError(f"No .txt files found in {story_dir}")

    logger.info(f"Found {len(story_files)} story files in {story_dir}")

    # Load each story
    stories = []
    failed_count = 0

    for story_path in story_files:
        try:
            # Read story content
            with open(story_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Skip empty files
            if not content:
                logger.warning(f"Skipping empty file: {story_path.name}")
                failed_count += 1
                continue

            # Create story dictionary
            story = {
                'name': story_path.stem,  # Filename without extension
                'path': str(story_path),
                'content': content
            }

            stories.append(story)
            logger.debug(f"Loaded: {story_path.name} ({len(content)} characters)")

        except Exception as e:
            logger.error(f"Failed to load {story_path.name}: {e}")
            failed_count += 1
            continue

    # Summary
    logger.info(f"Successfully loaded {len(stories)} stories")
    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count} stories")

    if len(stories) == 0:
        raise ValueError("No stories could be loaded successfully")

    return stories


def load_single_story(story_path: str) -> Dict[str, str]:
    """
    Load a single story file (used in Phase 2 for test stories).

    Args:
        story_path: Path to the story file

    Returns:
        Dictionary containing:
            {
                'name': str,      # Story name (filename without .txt)
                'path': str,      # Full path to the story file
                'content': str    # Story text content
            }

    Raises:
        FileNotFoundError: If story file doesn't exist
        ValueError: If story file is empty
    """
    path = Path(story_path)

    if not path.exists():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    logger.info(f"Loading story: {path.name}")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        if not content:
            raise ValueError(f"Story file is empty: {story_path}")

        story = {
            'name': path.stem,
            'path': str(path),
            'content': content
        }

        logger.info(f"Loaded story: {story['name']} ({len(content)} characters)")
        return story

    except Exception as e:
        logger.error(f"Failed to load story: {e}")
        raise