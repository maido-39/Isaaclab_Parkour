[INFO]: Time taken for simulation start : 5.322996 seconds
[INFO] Command Manager:  <CommandManager> contains 1 active terms.
+------------------------------------------------+
|              Active Command Terms              |
+-------+---------------+------------------------+
| Index | Name          |          Type          |
+-------+---------------+------------------------+
|   0   | base_velocity | UniformVelocityCommand |
+-------+---------------+------------------------+

[INFO] Event Manager:  <EventManager> contains 3 active terms.
+--------------------------------------+
| Active Event Terms in Mode: 'startup' |
+----------+---------------------------+
|  Index   | Name                      |
+----------+---------------------------+
|    0     | physics_material          |
|    1     | add_base_mass             |
+----------+---------------------------+
+---------------------------------------+
|  Active Event Terms in Mode: 'reset'  |
+--------+------------------------------+
| Index  | Name                         |
+--------+------------------------------+
|   0    | base_external_force_torque   |
|   1    | reset_base                   |
|   2    | reset_robot_joints           |
+--------+------------------------------+
+----------------------------------------------+
|    Active Event Terms in Mode: 'interval'    |
+-------+------------+-------------------------+
| Index | Name       | Interval time range (s) |
+-------+------------+-------------------------+
|   0   | push_robot |       (10.0, 15.0)      |
+-------+------------+-------------------------+

[INFO] Recorder Manager:  <RecorderManager> contains 0 active terms.
+---------------------+
| Active Recorder Terms |
+-----------+---------+
|   Index   | Name    |
+-----------+---------+
+-----------+---------+

[INFO] Action Manager:  <ActionManager> contains 1 active terms.
+------------------------------------+
|  Active Action Terms (shape: 12)   |
+--------+-------------+-------------+
| Index  | Name        |   Dimension |
+--------+-------------+-------------+
|   0    | joint_pos   |          12 |
+--------+-------------+-------------+

[INFO] Parkour Manager: <CommandManager> contains 1 active terms.
+-------------------------------------------+
|            Active Command Terms           |
+-------+--------------+--------------------+
| Index | Name         |        Type        |
+-------+--------------+--------------------+
|   0   | base_parkour | SimpleParkourEvent |
+-------+--------------+--------------------+

[INFO] Observation Manager: <ObservationManager> contains 1 groups.
+----------------------------------------------------------+
| Active Observation Terms in Group: 'policy' (shape: (180,)) |
+-----------+--------------------------------+-------------+
|   Index   | Name                           |    Shape    |
+-----------+--------------------------------+-------------+
|     0     | base_lin_vel                   |     (3,)    |
|     1     | base_ang_vel                   |     (3,)    |
|     2     | projected_gravity              |     (3,)    |
|     3     | velocity_commands              |     (3,)    |
|     4     | joint_pos                      |    (12,)    |
|     5     | joint_vel                      |    (12,)    |
|     6     | actions                        |    (12,)    |
|     7     | height_scan                    |    (132,)   |
+-----------+--------------------------------+-------------+

[INFO] Termination Manager:  <TerminationManager> contains 2 active terms.
+---------------------------------+
|     Active Termination Terms    |
+-------+--------------+----------+
| Index | Name         | Time Out |
+-------+--------------+----------+
|   0   | time_out     |   True   |
|   1   | base_contact |  False   |
+-------+--------------+----------+

[INFO] Reward Manager:  <RewardManager> contains 10 active terms.
+-----------------------------------------+
|           Active Reward Terms           |
+-------+----------------------+----------+
| Index | Name                 |   Weight |
+-------+----------------------+----------+
|   0   | track_lin_vel_xy_exp |      1.5 |
|   1   | track_ang_vel_z_exp  |     0.75 |
|   2   | lin_vel_z_l2         |     -2.0 |
|   3   | ang_vel_xy_l2        |    -0.05 |
|   4   | dof_torques_l2       |  -0.0002 |
|   5   | dof_acc_l2           | -2.5e-07 |
|   6   | action_rate_l2       |    -0.01 |
|   7   | feet_air_time        |     0.01 |
|   8   | flat_orientation_l2  |      0.0 |
|   9   | dof_pos_limits       |      0.0 |
+-------+----------------------+----------+

[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+----------------------+
| Active Curriculum Terms |
+-----------+----------+
|   Index   | Name     |
+-----------+----------+
+-----------+----------+

Creating window for environment.
[INFO]: Completed setting up the environment...
/home/syaro/miniconda3/envs/isc-pak/lib/python3.10/site-packages/torch/nn/init.py:511: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")
Actor MLP: Actor(
  (priv_encoder): Sequential(
    (0): Linear(in_features=0, out_features=64, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=64, out_features=20, bias=True)
    (3): ELU(alpha=1.0)
  )
  (history_encoder): StateHistoryEncoder(
    (activation_fn): ELU(alpha=1.0)
    (encoder): Sequential(
      (0): Linear(in_features=48, out_features=30, bias=True)
      (1): ELU(alpha=1.0)
    )
    (conv_layers): Sequential(
      (0): Conv1d(30, 20, kernel_size=(4,), stride=(2,))
      (1): ELU(alpha=1.0)
      (2): Conv1d(20, 10, kernel_size=(2,), stride=(1,))
      (3): ELU(alpha=1.0)
      (4): Flatten(start_dim=1, end_dim=-1)
    )
    (linear_output): Sequential(
      (0): Linear(in_features=30, out_features=20, bias=True)
      (1): ELU(alpha=1.0)
    )
  )
  (scan_encoder): Sequential(
    (0): Linear(in_features=132, out_features=128, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=128, out_features=64, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=64, out_features=32, bias=True)
    (5): Tanh()
  )
  (actor_backbone): Sequential(
    (0): Linear(in_features=100, out_features=512, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=512, out_features=256, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
)
Critic MLP: Sequential(
  (0): Linear(in_features=180, out_features=512, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=512, out_features=256, bias=True)
  (3): ELU(alpha=1.0)
  (4): Linear(in_features=256, out_features=128, bias=True)
  (5): ELU(alpha=1.0)
  (6): Linear(in_features=128, out_features=1, bias=True)
)
estimator MLP: DummyEstimator()
Updating dagger...
