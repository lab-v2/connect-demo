"""
PyReason Runner Module

Runs PyReason with ground atoms and learned rules to generate trace files.
The trace files are used by the abduction algorithm to determine which features to improve.
"""

import re
import logging
import shutil
from pathlib import Path
from typing import Dict, List

try:
    import networkx as nx
except ImportError:
    raise ImportError("networkx is required. Install with: pip install networkx")

try:
    import pyreason as pr
except ImportError:
    raise ImportError("pyreason is required. Install with: pip install pyreason")

logger = logging.getLogger(__name__)


# ==========================================================
# Name Conversion (same as graph_builder)
# ==========================================================

def to_atom_id(name: str) -> str:
    """
    Convert story name to atom ID format.

    Args:
        name: Story name

    Returns:
        Atom ID (lowercase, underscores)
    """
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


# ==========================================================
# NetworkX Graph Creation
# ==========================================================

def create_networkx_graph(ground_atoms: Dict[str, List[float]]) -> nx.DiGraph:
    """
    Create NetworkX graph from ground atoms.

    The graph structure is needed by PyReason but is not saved to disk.

    Args:
        ground_atoms: Dict of ground atoms

    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()

    # Extract nodes from individualistic_feature atoms
    for atom in ground_atoms:
        if atom.startswith('individualistic_feature('):
            # Parse: individualistic_feature(story, feature)
            content = atom[len('individualistic_feature('):-1]
            parts = content.split(', ', 1)
            if len(parts) == 2:
                story, feature = parts

                # Clean node names (remove spaces)
                story_node = story.replace(' ', '')
                feature_node = feature.replace(' ', '')

                # Add nodes with attributes
                G.add_node(story_node, story_name='1,1')
                G.add_node(feature_node, **{feature.replace(' ', '_'): '1,1'})

    logger.debug(f"Created NetworkX graph with {G.number_of_nodes()} nodes")
    return G


# ==========================================================
# PyReason Execution
# ==========================================================

def run_pyreason(
        ground_atoms: Dict[str, List[float]],
        rules_file: str,
        story_name: str,
        output_dir: str
) -> str:
    """
    Run PyReason with ground atoms and learned rules.

    Args:
        ground_atoms: Dict of ground atoms from graph_builder
        rules_file: Path to learned rules .txt file
        story_name: Name of the story
        output_dir: Directory to save trace files (iteration directory)

    Returns:
        Path to trace directory containing CSV files
    """
    logger.info("Running PyReason...")

    # Create trace directory (clean slate)
    trace_dir = Path(output_dir) / "pyreason_traces"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
        logger.debug(f"Removed old trace directory: {trace_dir}")

    trace_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created trace directory: {trace_dir}")

    # Configure PyReason settings
    pr.settings.atom_trace = True
    pr.settings.save_graph_attributes_to_trace = True
    logger.debug("PyReason settings configured")

    # Create NetworkX graph
    logger.debug("Creating NetworkX graph...")
    G = create_networkx_graph(ground_atoms)

    # Load graph into PyReason
    logger.debug("Loading graph into PyReason...")
    pr.load_graph(G)

    # Add facts from ground atoms
    logger.debug("Adding facts to PyReason...")
    fact_count = 0

    for atom, conf in ground_atoms.items():
        if atom.startswith('individualistic_feature('):
            # Parse: individualistic_feature(story, feature)
            content = atom[len('individualistic_feature('):-1]
            parts = content.split(', ', 1)
            if len(parts) == 2:
                story, feature = parts
                c = conf[0]  # Get confidence value

                # Add fact
                pr.add_fact(pr.Fact(
                    fact_text=f'individualistic_feature({story},{feature}) : [{c},{c}]',
                    name=f'fact_{story}_{feature}',
                    static=True
                ))
                fact_count += 1

    logger.info(f"Added {fact_count} facts to PyReason")

    # Load rules from file
    logger.debug(f"Loading rules from: {rules_file}")
    pr.add_rules_from_file(str(rules_file))
    logger.info("Rules loaded successfully")

    # Run reasoning
    logger.info("Running PyReason reasoning (1 timestep)...")
    interpretation = pr.reason(timesteps=1)
    logger.info("Reasoning completed")

    # Save rule traces
    logger.info(f"Saving rule traces to: {trace_dir}")
    pr.save_rule_trace(interpretation=interpretation, folder=str(trace_dir))

    # Verify trace files were created
    trace_files = list(trace_dir.glob("rule_trace_*.csv"))
    if len(trace_files) == 0:
        logger.warning("No trace files generated by PyReason")
    else:
        logger.info(f"✓ Generated {len(trace_files)} trace files:")
        for trace_file in trace_files:
            logger.info(f"  - {trace_file.name}")

    return str(trace_dir)


# ==========================================================
# For Testing/Direct Execution
# ==========================================================

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 4:
        print("Usage: python pyreason_runner.py <ground_atoms.json> <rules.txt> <story_name>")
        print("Example: python pyreason_runner.py iteration_0/ground_atoms.json rules.txt 'Community Time'")
        sys.exit(1)

    ground_atoms_file = sys.argv[1]
    rules_file = sys.argv[2]
    story_name = sys.argv[3]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load ground atoms
    with open(ground_atoms_file, 'r') as f:
        ground_atoms = json.load(f)

    # Run PyReason
    trace_dir = run_pyreason(
        ground_atoms=ground_atoms,
        rules_file=rules_file,
        story_name=story_name,
        output_dir="."
    )

    print(f"\n✓ PyReason traces saved to: {trace_dir}")