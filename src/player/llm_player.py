from src.player.gpt_player import LLMPlayer
from src.environment.abstract_battle import AbstractBattle
import json
from peft import PeftModel
import transformers
import torch
from player import BattleOrder
from tqdm import tqdm

my_token = "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"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

class LLAMAPlayer(LLMPlayer):

    def __init__(self, battle_format,
                 model_name_or_path: str = "",
                 # tokenizer_path: str = "",
                 lora_weights: str = "",
                 model_max_length: int = 2048,
                 w_reason = False,
                 log_dir = "",
                 account_configuration=None,
                 server_configuration=None,
                 ):
        super().__init__(battle_format=battle_format,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)
        # initialize the LLAMA model
        # load the LLAMA model

        self.except_cnt = 0
        self.total_cnt = 0
        self.log_dir = log_dir
        self.w_reason = w_reason
        self.last_output = None
        self.last_state_prompt = None

        assert (model_name_or_path), "Please specify the model path"

        # if tokenizer_path:
        #     self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #         tokenizer_path,
        #         use_fast=False,
        #         use_auth_token=my_token
        #     )
        # else:

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
            use_auth_token=my_token
        )

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=my_token
        )

        if lora_weights:
            print("Recover LoRA weights..")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
            )

        print("Loading finished...")
        self.model.eval()

    def choose_move(self, battle: AbstractBattle):

        if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            next_action = BattleOrder(battle.available_switches[0])
            return next_action

        # state_prompt = self.state_translate(battle)
        system_prompt, state_prompt = self.state_translate(battle) # add lower case

        if battle.active_pokemon.fainted:

            constraint_prompt1 = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt2 = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            # constraint_prompt3 = '''Generate top-k (k<=3) best switch options, evaluate the possible consequences of each option. Pick the best option based on the evaluation. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>","evaluation":"<evaluation>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>","evaluation":"<evaluation>"},"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt3 = '''Generate top-k (k<=3) best switch options and pick out the best option by considering their consequences. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"},"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt4_1 = '''Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt4_2 = '''Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''

        else:
            constraint_prompt1 = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt2 = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            # constraint_prompt3 = '''Generate top-k (k<=3) best action options, and evaluate their consequences in terms of speed, type advantage/weakness, and the first-move hit if switch-in. Pick out the best action based on the evaluation. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>", "evaluation":"<evaluation>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>", "evaluation":"<evaluation>"}, "decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt3 = '''Generate top-k (k<=3) best action options and pick out the best action by considering their consequences. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, "decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt4_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt4_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}\n'''

        state_prompt1 = state_prompt + constraint_prompt1
        state_prompt2 = state_prompt + constraint_prompt2
        state_prompt3 = state_prompt + constraint_prompt3
        state_prompt4_1 = state_prompt + constraint_prompt4_1
        state_prompt4_2 = state_prompt + constraint_prompt4_2

        user_prompt = system_prompt + state_prompt1 + 'Output:{"'
        print("===================")
        print(user_prompt)

        input_dict = self.tokenizer(user_prompt, return_tensors="pt").to("cuda")
        input_ids = input_dict["input_ids"]

        next_action = None
        for i in range(5):
            try:
                with torch.no_grad():
                    generation_output = self.model.generate(
                                        inputs=input_ids,
                                        temperature=0.8,
                                        do_sample=True,
                                        num_beams=1,
                                        max_new_tokens=100,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.eos_token_id
                                        # eos_token_id=self.tokenizer.eos_token_id,
                                        # pad_token_id=self.tokenizer.pad_token_id # self.tokenizer.pad_token_id
                                    )
                s = generation_output[0]
                llm_output = self.tokenizer.decode(s, skip_special_tokens=True)
                llm_output = llm_output.split("Output:")[1]
                next_action, _ = self.parse(llm_output, battle)
            except Exception as e:
                continue

        if next_action:
            print("LLM output:", llm_output)
            with open(f"{self.log_dir}/output.jsonl", "a") as f:
                f.write(json.dumps({"prompt": user_prompt, "llm_output": llm_output}) + "\n")
        else:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
            print("Exception occured.....")

        self.total_cnt += 1
        return next_action
        # except Exception as e:
        #     continue

    def choose_move_reward(self, battle: AbstractBattle):

        state_prompt = self.state_translate(battle).lower()
        if self.w_reason == False:
            state_prompt += "Do not output any explanation or reasoning. "
        state_prompt += 'Output:{"'
        print("===================")
        print(state_prompt)

        # check last action is valid
        if self.last_output and self.last_state_prompt:

            # calculate
            # reward = self.reward_computing_helper()
            reward = self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0)

            # dump logs
            last_log = {"prompt": self.last_state_prompt,
                        "llm_output": self.last_output,
                        "reward": reward}


        input_dict = self.tokenizer(state_prompt, return_tensors="pt").to("cuda")
        input_ids = input_dict["input_ids"]

        next_action = None
        for i in range(5):
            try:
                with torch.no_grad():
                    generation_output = self.model.generate(
                                        inputs=input_ids,
                                        # top_p=1,
                                        # do_sample=False,
                                        temperature=0.8,
                                        do_sample=True,
                                        num_beams=1,
                                        max_new_tokens=500,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.eos_token_id
                                        # eos_token_id=self.tokenizer.eos_token_id,
                                        # pad_token_id=self.tokenizer.pad_token_id # self.tokenizer.pad_token_id
                                    )
                s = generation_output[0]
                llm_output = self.tokenizer.decode(s, skip_special_tokens=True)
                llm_output = llm_output.split("Output:")[1]
                next_action = self.parse(llm_output, battle)
            except Exception as e:
                continue

        if next_action:
            print("LLM output:", llm_output)
            self.last_output = llm_output

        else:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
            print("Exception occured.....")

            self.last_output = None

        self.total_cnt += 1
        self.last_state_prompt = state_prompt
        return next_action
        # except Exception as e:
        #     continue
