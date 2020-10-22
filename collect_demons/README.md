# Data Collection

This module provides a pit-stop to collect trajectories. The `collect_by` config defined in `demons_config` specifies how you want to collect the data - by teleopetation, a specific/random policy or an imitation-based policy trained on the data. The imitation policy is defined by a RNN-based gaussian policy in `imitate_play.py`. Since teleoperation and collection of random trajectories is environment specific, they are defined in the environment's corresponding DEG. But, they should be run from the `main.py` here. 
