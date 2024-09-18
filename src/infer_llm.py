import asyncio
from tqdm import tqdm
import numpy as np
import os
import hydra
import pickle as pkl
import argparse
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Set
from src.utils.llm_utils import get_local_dir, get_local_run_dir
from client.account_configuration import AccountConfiguration

from src.player import LLMPlayer, HeuristicsPlayer

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


@hydra.main(version_base=None, config_path="config", config_name="config")
async def main(config: DictConfig):

    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    print(OmegaConf.to_yaml(config))
    heuristic_player = HeuristicsPlayer(battle_format="gen8randombattle")

    save_replay_dir = os.path.join("./battle_log/", config.exp_name)
    os.makedirs(save_replay_dir, exist_ok=True)

    llm_player = LLMPlayer(config=config,
                           save_replay_dir=save_replay_dir,
                           account_configuration=AccountConfiguration("test_player926", "123456"),
                           )

    # dynamax is disabled for local battles.
    heuristic_player._dynamax_disable = True
    llm_player._dynamax_disable = True

    # play against bot for five battles
    for i in tqdm(range(5)):
        x = np.random.randint(0, 100)
        if x > 50:
            await heuristic_player.battle_against(llm_player, n_battles=1)
        else:
            await llm_player.battle_against(heuristic_player, n_battles=1)
        for battle_id, battle in llm_player.battles.items():
            with open(f"{save_replay_dir}/{battle_id}.pkl", "wb") as f:
                pkl.dump(battle, f)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
