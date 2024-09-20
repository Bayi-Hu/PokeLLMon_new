from typing import List
import json
import os

from src.environment.abstract_battle import AbstractBattle
from src.environment.double_battle import DoubleBattle
from src.environment.move_category import MoveCategory
from src.environment.pokemon import Pokemon
from src.environment.side_condition import SideCondition
from src.player.player import Player
from src.data.gen_data import GenData
from src.player.battle_order import BattleOrder


def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

    move_type_damage_multiplier_list = []

    if type_2:
        for type in TYPE_list:
            move_type_damage_multiplier_list.append(type_chart[type_1][type] * type_chart[type_2][type])
        move_type_damage_multiplier_dict = dict(zip(TYPE_list, move_type_damage_multiplier_list))
    else:
        move_type_damage_multiplier_dict = type_chart[type_1]

    effective_type_list = []
    extreme_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    for type, value in move_type_damage_multiplier_dict.items():
        if value == 2:
            effective_type_list.append(type)
        elif value == 4:
            extreme_type_list.append(type)
        elif value == 1 / 2:
            resistant_type_list.append(type)
        elif value == 1 / 4:
            extreme_resistant_type_list.append(type)
        elif value == 0:
            immune_type_list.append(type)
        else:  # value == 1
            continue

    if constraint_type_list:
        extreme_type_list = list(set(extreme_type_list).intersection(set(constraint_type_list)))
        effective_type_list = list(set(effective_type_list).intersection(set(constraint_type_list)))
        resistant_type_list = list(set(resistant_type_list).intersection(set(constraint_type_list)))
        extreme_resistant_type_list = list(set(extreme_resistant_type_list).intersection(set(constraint_type_list)))
        immune_type_list = list(set(immune_type_list).intersection(set(constraint_type_list)))

    return extreme_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list


def move_type_damage_wraper(pokemon_name, type_1, type_2, type_chart, constraint_type_list=None):

    move_type_damage_prompt = ""
    extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list)

    if effective_type_list or resistant_type_list or immune_type_list:

        move_type_damage_prompt = f"{pokemon_name}"
        if extreme_effective_type_list:
            move_type_damage_prompt = move_type_damage_prompt + " can be super-effectively attacked by " + ", ".join(
                extreme_effective_type_list) + " moves"
        if effective_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", can be effectively attacked by " + ", ".join(
                effective_type_list) + " moves"
        if resistant_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is resistant to " + ", ".join(
                resistant_type_list) + " moves"
        if extreme_resistant_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is super-resistant to " + ", ".join(
                extreme_resistant_type_list) + " moves"
        if immune_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is immuned to " + ", ".join(
                immune_type_list) + " moves"

    return move_type_damage_prompt


class RandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self.choose_random_move(battle)


class MaxBasePowerPlayer(Player):
    def choose_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)

class HeuristicsPlayer(Player):
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_dynamax(self, battle: AbstractBattle, n_remaining_mons: int):
        if battle.can_dynamax and self._dynamax_disable is False:
            # Last full HP mon
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # Matchup advantage and full hp on full hp
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False

    def _should_switch_out(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        # If there is a decent switch in...
        if [
            m
            for m in battle.available_switches
            if self._estimate_matchup(m, opponent) > 0
        ]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                self._estimate_matchup(active, opponent)
                < self.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        return False

    def _stat_estimation(self, mon: Pokemon, stat: str):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    def calc_reward(
            self, current_battle: AbstractBattle
    ) -> float:
        # Calculate the reward
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def choose_move(self, battle: AbstractBattle):
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)

        self.gen = GenData.from_format(self.format)
        for mon in battle.team.values():
            self.move_set = self.move_set.union(set(mon.moves.keys()))
            self.item_set.add(mon.item)
            self.ability_set.add(mon.ability)
            try:
                self.pokemon_item_dict[mon.species].add(mon.item)
            except:
                self.pokemon_item_dict[mon.species] = set()
                self.pokemon_item_dict[mon.species].add(mon.item)
            try:
                self.pokemon_ability_dict[mon.species].add(mon.ability)
            except:
                self.pokemon_ability_dict[mon.species] = set()
                self.pokemon_ability_dict[mon.species].add(mon.ability)
            for name, move in mon.moves.items():
                try:
                    self.pokemon_move_dict[mon.species][name][3] += 1
                except:
                    try:
                        self.pokemon_move_dict[mon.species][name] = [name, move.type.name, move.base_power, 1]
                    except:
                        self.pokemon_move_dict[mon.species] = {}
                        self.pokemon_move_dict[mon.species][name] = [name, move.type.name, move.base_power, 1]

        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
            opponent, "def"
        )
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
            opponent, "spd"
        )

        next_action = None
        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )
            n_opp_remaining_mons = 6 - len(
                [m for m in battle.opponent_team.values() if m.fainted is True]
            )

            # Entry hazard...
            for move in battle.available_moves:
                # ...setup
                if (
                    n_opp_remaining_mons >= 3
                    and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions
                ):
                    next_action = self.create_order(move)
                    break

                # ...removal
                elif (
                    battle.side_conditions
                    and move.id in self.ANTI_HAZARDS_MOVES
                    and n_remaining_mons >= 2
                ):
                    next_action = self.create_order(move)
                    break

            # Setup moves
            if (
                next_action is None
                and active.current_hp_fraction == 1
                and self._estimate_matchup(active, opponent) > 0
            ):
                for move in battle.available_moves:
                    if (
                        self._boost_disable is False
                        and move.boosts
                        and sum(move.boosts.values()) >= 2
                        and move.target == "self"
                        and min(
                            [active.boosts[s] for s, v in move.boosts.items() if v > 0]
                        )
                        < 6
                    ):
                        next_action = self.create_order(move)
                        break

            if next_action is None:
                move = max(
                    battle.available_moves,
                    key=lambda m: m.base_power
                    * (1.5 if m.type in active.types else 1)
                    * (
                        physical_ratio
                        if m.category == MoveCategory.PHYSICAL
                        else special_ratio
                    )
                    * m.accuracy
                    * m.expected_hits
                    * opponent.damage_multiplier(m),
                )
                next_action = self.create_order(
                    move, dynamax=self._should_dynamax(battle, n_remaining_mons)
                )

        if next_action is None and battle.available_switches:
            switches: List[Pokemon] = battle.available_switches
            next_action = self.create_order(
                max(
                    switches,
                    key=lambda s: self._estimate_matchup(s, opponent),
                )
            )

        if next_action:
            # action = next_action.message.split(" ")[1]
            # object = next_action.message.split(" ")[2]
            #
            # if action == "switch":
            #     dump_log.update({"output": '{"' + action + '": "' + object + '"}'})
            # if action == "move":
            #     dump_log.update(
            #         {"output": '{"' + action + '": "' + object + '", "dynamax": "' + str(next_action.dynamax) + '"}'})
            #
            # dump_log_dir = "/Users/husihao/Documents/PokemonProject/PokeLLMon/battle_log"
            # if dump_log_dir:
            #     with open(os.path.join(dump_log_dir, "heuristic_battle_log.jsonl"), "a") as f:
            #         f.write(json.dumps(dump_log) + "\n")
            pass

        else:
            next_action = self.choose_random_move(battle)

        return next_action
