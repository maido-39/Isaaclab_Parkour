from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.extreme_parkour import * 

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
#     size=(16.0, 4.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=40,
#     horizontal_scale=0.08, ## original scale is 0.05, But Computing issue in IsaacLab see this issue in https://github.com/isaac-sim/IsaacLab/issues/2187
#     vertical_scale=0.005,
#     slope_threshold=1.5,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     curriculum= True,
#     sub_terrains={
#         "parkour_gap": ExtremeParkourGapTerrainCfg(
#                         proportion=0.2,
#                         apply_roughness=True,
#                         x_range = (0.8, 1.5),
#                         half_valid_width = (0.6, 1.2),
#                         gap_size = '0.1 + 0.7*difficulty'
#                         ),
#         "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
#                         proportion=0.2,
#                         apply_roughness=True,
#                         x_range = (1.2, 2.2),
#                         half_valid_width = (0.4,0.8),
#                         hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.25*difficulty'
#                         ),
#         "parkour_flat": ExtremeParkourHurdleTerrainCfg(
#                         proportion=0.2,
#                         apply_roughness=True,
#                         apply_flat=True,
#                         x_range = (1.2, 2.2),
#                         half_valid_width = (0.4,0.8),
#                         hurdle_height_range= '0.1+0.1*difficulty, 0.15+0.15*difficulty'
#                         ),
#         "parkour_step": ExtremeParkourStepTerrainCfg(
#                         proportion=0.2,
#                         apply_roughness=True,
#                         x_range = (0.3,1.5),
#                         half_valid_width = (0.5, 1),
#                         step_height = '0.1 + 0.35*difficulty'
#                         ),
#         "parkour": ExtremeParkourTerrainCfg(
#                         proportion=0.2,
#                         apply_roughness=True,
#                         x_range  = '-0.1, 0.1+0.3*difficulty',
#                         y_range  = '0.2, 0.3+0.1*difficulty',
#                         stone_len  = '0.9 - 0.3*difficulty, 1 - 0.2*difficulty',
#                         incline_height = '0.25*difficulty',
#                         last_incline_height = 'incline_height + 0.1 - 0.1*difficulty'
#                         ),
#         "parkour_demo": ExtremeParkourDemoTerrainCfg(
#                         proportion=0.0,
#                         apply_roughness=True,
#                         ),

#     },
# )


# Use the existing rough terrain config from IsaacLab's terrain configs.
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG as _ROUGH_TERRAINS_CFG
import copy
import numpy as np

# Convert the generic ROUGH_TERRAINS_CFG (TerrainGeneratorCfg) into a
# ParkourTerrainGeneratorCfg so parkour-specific code (which expects fields
# like `num_goals`) works. We copy the common fields and provide sensible
# defaults when the source config doesn't have them.

# Prepare wrapped sub_terrains where each callable accepts (difficulty, cfg, num_goals)
# by forwarding to the original 2-argument functions.
_original_sub_terrains = getattr(_ROUGH_TERRAINS_CFG, "sub_terrains", {})
_wrapped_sub_terrains = {}
for _name, _sub_cfg in _original_sub_terrains.items():
	try:
		_sub_copy = copy.copy(_sub_cfg)
	except Exception:
		# Fallback: if copy fails, use the original object (still safe to
		# replace the callable attribute)
		_sub_copy = _sub_cfg

	_orig_fn = getattr(_sub_copy, "function", None)
	if _orig_fn is not None:
		def _make_wrapper(fn):
			def _wrapper(difficulty, cfg, num_goals):
				# Try calling a 3-arg function first (some newer functions accept num_goals)
				try:
					res = fn(difficulty, cfg, num_goals)
				except TypeError:
					# Fallback to legacy 2-arg signature
					res = fn(difficulty, cfg)

				# If the function already returned the full 5-tuple, return it
				if isinstance(res, tuple) and len(res) == 5:
					# Ensure goals are (N,2) as expected by ParkourTerrainGenerator
					meshes, origin, goals, goal_heights, x_edge_mask = res
					if hasattr(goals, "shape") and getattr(goals, "ndim", 0) == 2 and goals.shape[1] >= 2:
						goals = goals[:, :2]
					return meshes, origin, goals, goal_heights, x_edge_mask

				# If it returned at least (meshes, origin), synthesize the remaining
				if isinstance(res, tuple) and len(res) >= 2:
					meshes, origin = res[0], res[1]
					# Default goals placed at origin (num_goals x 2) -> (x, y)
					goals = np.zeros((int(num_goals), 2), dtype=float)
					# Default goal heights and x_edge_mask
					goal_heights = np.zeros((int(num_goals),), dtype=np.int16)
					try:
						width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
						length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
					except Exception:
						width_pixels = 2
						length_pixels = 2
					x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=np.int16)
					return meshes, origin, goals, goal_heights, x_edge_mask
				
				# Unexpected return value
				raise ValueError("Unexpected return value from sub-terrain function")

			return _wrapper

		_sub_copy.function = _make_wrapper(_orig_fn)

	_wrapped_sub_terrains[_name] = _sub_copy

EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
	seed=getattr(_ROUGH_TERRAINS_CFG, "seed", None),
	curriculum=getattr(_ROUGH_TERRAINS_CFG, "curriculum", False),
	size=getattr(_ROUGH_TERRAINS_CFG, "size", (8.0, 8.0)),
	border_width=getattr(_ROUGH_TERRAINS_CFG, "border_width", 0.0),
	border_height=getattr(_ROUGH_TERRAINS_CFG, "border_height", 1.0),
	num_rows=getattr(_ROUGH_TERRAINS_CFG, "num_rows", 1),
	num_cols=getattr(_ROUGH_TERRAINS_CFG, "num_cols", 1),
	color_scheme=getattr(_ROUGH_TERRAINS_CFG, "color_scheme", "none"),
	horizontal_scale=getattr(_ROUGH_TERRAINS_CFG, "horizontal_scale", 0.1),
	vertical_scale=getattr(_ROUGH_TERRAINS_CFG, "vertical_scale", 0.005),
	slope_threshold=getattr(_ROUGH_TERRAINS_CFG, "slope_threshold", 0.75),
	sub_terrains=_wrapped_sub_terrains,
	difficulty_range=getattr(_ROUGH_TERRAINS_CFG, "difficulty_range", (0.0, 1.0)),
	use_cache=getattr(_ROUGH_TERRAINS_CFG, "use_cache", False),
	cache_dir=getattr(_ROUGH_TERRAINS_CFG, "cache_dir", "/tmp/isaaclab/terrains"),
	# Parkour-specific fields (have defaults in ParkourTerrainGeneratorCfg)
	num_goals=getattr(_ROUGH_TERRAINS_CFG, "num_goals", 8),
	terrain_names=getattr(_ROUGH_TERRAINS_CFG, "terrain_names", []),
	random_difficulty=getattr(_ROUGH_TERRAINS_CFG, "random_difficulty", False),
)