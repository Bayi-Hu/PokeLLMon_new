import pickle as pkl
import numpy as np


with open("./battle_log/llama2_7bchat_temperature1.2_T6/all_battles.pkl", "rb") as f:
    all_battles = pkl.load(f)

beat_list = []
remain_list = []
win_list = []
tag_list = []
turn_list = []

for tag, battle in all_battles.items():
    if battle.finished:
        beat_score = 0
        for mon in battle.opponent_team.values():
            beat_score += (1 - mon.current_hp_fraction)

        beat_list.append(beat_score)
        remain_score = 0

        for mon in battle.team.values():
            remain_score += mon.current_hp_fraction
        remain_list.append(remain_score)
        if battle.won:
            win_list.append(1)
        tag_list.append(tag)
        turn_list.append(battle.turn)

beat_list = np.array(beat_list)
remain_list = np.array(remain_list)
print("battle #", len(beat_list))
print("battle score:", np.mean(beat_list)+np.mean(remain_list))
print("beat num:", np.mean(beat_list))
print("remain num:", np.mean(remain_list))
print("win rate:", len(win_list)/len(beat_list))
print("turn:", np.mean(turn_list))
