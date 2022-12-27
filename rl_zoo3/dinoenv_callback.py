from stable_baselines3.common.callbacks import BaseCallback
from gymdinoenv.envs import DinoEnv
class DinoEnvCallback(BaseCallback):
    def _on_step(self) -> bool:
        env: DinoEnv = self.training_env
        done: bool = self.locals['dones'][0]
        info: dict = self.locals['infos'][0]
        score: int = info['score']
        if done:
            self.logger.record_mean('ep_score_mean', score)
        return True