import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import weed_extraction  # noqa: F401  (registers WeedExtraction-v0)

SMALL = dict(width=160, height=90)


def test_passes_gymnasium_checker():
    env = gym.make("WeedExtraction-v0", **SMALL)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_observations():
    env = gym.make("WeedExtraction-v0", cameras=["d435i_color", "d435i_depth"], **SMALL)
    obs, _ = env.reset(seed=0)
    assert obs["d435i_color"].shape == (90, 160, 3) and obs["d435i_color"].mean() > 10
    # At home the wrist camera looks down at the walkway ~0.6 m below.
    assert 0.3 < np.median(obs["d435i_depth"]) < 1.0
    _, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["uprooted"] == []
    env.close()
