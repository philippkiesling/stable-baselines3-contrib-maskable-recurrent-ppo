import unittest
import os
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.utils import get_action_masks
from maskable_recurrent.common import evaluate_policy
from maskable_recurrent.ppo_mask_recurrent import MaskableRecurrentPPO
from maskable_recurrent.ppo_mask_recurrent import MaskableRecurrentActorCriticPolicy


class TestMyFunction(unittest.TestCase):

    def test_discrete_observation_space(self):
        """
        Test that the model can learn, evaluate, predict with a discrete observation space.
        Test save and load functions for discrete observation space.
        """
        env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
        model = MaskableRecurrentPPO(MaskableRecurrentActorCriticPolicy, env, gamma=0.4, seed=32, verbose = -1)
        model.learn(100)

        evaluate_policy(model, env, n_eval_episodes=5, warn=False)

        model.save("ppo_mask")

        del model  # remove to demonstrate saving and loading
        model = MaskableRecurrentPPO.load("ppo_mask")
        # Remove File
        os.remove("ppo_mask.zip")

        obs = env.reset()
        for i in range(100):
            # Retrieve current action mask
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, rewards, dones, info = env.step(action)

if __name__ == '__main__':
    unittest.main()
