"""
This file contains the custom callbacks for the battery trading environment.
This currently includes the EvalCallbackActionMask, which is a custom evaluation callback for envs with action masking.
The EvalCallbackActionMask is based on the original EvalCallback from stable_baselines3. In contrast to the original EvalCallback, the EvalCallbackActionMask applies the action mask during evaluation.
"""
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_policy_mlp
from maskable_recurrent.common.evaluation import evaluate_policy as evaluate_policy_recurrent
from stable_baselines3.common.vec_env import sync_envs_normalization
import numpy as np
import os

class EvalCallbackActionMask(EvalCallback):
    """
    Custom evaluation callback for envs with action masking.
    The original EvalCallback does not apply the action mask during evaluation.
    This callback applies the action mask during evaluation.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the Environment

        Args:
            max_charge: Maximum charge/discharge rate
            total_storage_capacity: Total Storage Capacity of the Battery
            initial_charge: Initial SOC of the Battery
            max_SOC: Maximum SOC of the Battery
            price_time_horizon:
            data_root_path: Path to the data
            time_interval: time per step (15min, 30min, H)
            wandb_run: Wandb run to log the data
            n_past_timesteps: number of past day_ahead_steps to use as input
            time_features: Use time features
            day_ahead_environment: -
            prediction_output: We can either predict the action or the future SOC, or percentage of potential charge/discharge
            max_steps: Maximum number of steps in the environment
            reward_shaping: Reward shaping schedule, None, "linear", "constant", "cosine", "exponential"
            cumulative_coef: Coefficient for the cumulative reward shaping
            eval_env: If the environment is used for evaluation (Evaluation Environment is always one step ahead of the training environment)
            n_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
        """

        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy_mlp(
                self.model,
                self.eval_env,
                #use_masking = self.use_masking,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class EvalCallbackRecurrentActionMask(EvalCallback):
    """
    Custom evaluation callback for envs with action masking.
    The original EvalCallback does not apply the action mask during evaluation.
    This callback applies the action mask during evaluation.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the Environment

        Args:
            max_charge: Maximum charge/discharge rate
            total_storage_capacity: Total Storage Capacity of the Battery
            initial_charge: Initial SOC of the Battery
            max_SOC: Maximum SOC of the Battery
            price_time_horizon:
            data_root_path: Path to the data
            time_interval: time per step (15min, 30min, H)
            wandb_run: Wandb run to log the data
            n_past_timesteps: number of past day_ahead_steps to use as input
            time_features: Use time features
            day_ahead_environment: -
            prediction_output: We can either predict the action or the future SOC, or percentage of potential charge/discharge
            max_steps: Maximum number of steps in the environment
            reward_shaping: Reward shaping schedule, None, "linear", "constant", "cosine", "exponential"
            cumulative_coef: Coefficient for the cumulative reward shaping
            eval_env: If the environment is used for evaluation (Evaluation Environment is always one step ahead of the training environment)
            n_steps: Number of steps in the environment, needed for evaluation Environments (No influence on training environments)
            gaussian_noise: Add gaussian noise to the day ahead prices and intra day prices
            noise_std: Standard deviation of the gaussian noise
        """

        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy_recurrent(
                self.model,
                self.eval_env,
                #use_masking = self.use_masking,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
