# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable

from isaaclab.utils import configclass
from isaaclab.actuators.actuator_cfg import DCMotorCfg, ActuatorNetMLPCfg
from . import parkour_actuator_pd

@configclass
class ParkourDCMotorCfg(DCMotorCfg):

    class_type: type = parkour_actuator_pd.ParkourDCMotor

    saturation_effort: dict[str, float] | None = None
    """Peak motor force/torque of the electric DC motor (in N-m)."""


@configclass  
class ParkourActuatorNetMLPCfg(ActuatorNetMLPCfg):
    """Configuration for Go1 MLP-based actuator model for parkour."""
    
    # Use Isaac Lab's standard Go1 actuator network
    # This will be set in the scene configuration
