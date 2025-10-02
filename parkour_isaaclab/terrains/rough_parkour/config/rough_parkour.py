"""Configuration for rough terrain adapted for parkour system."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg

# Create a rough terrain configuration compatible with the parkour system
ROUGH_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    # Use similar size and structure as the original parkour config but simpler
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,  # Disable curriculum since rough terrain doesn't need it
    num_goals=1,  # Minimal goals since we're removing subgoal system
    terrain_names=[],
    random_difficulty=False,
    # Use the same sub-terrains as the original rough terrain config
    sub_terrains=ROUGH_TERRAINS_CFG.sub_terrains,
)
"""Rough terrain configuration adapted for parkour system."""
