from src.player.gpt_player import GPTPlayer
from src.environment.abstract_battle import AbstractBattle
from src.utils.llm_utils import disable_dropout, get_local_dir
import json
import transformers
import torch
from player import BattleOrder


class LLMPlayer(GPTPlayer):
    def __init__(self,
                 save_replay_dir="",
                 account_configuration=None,
                 server_configuration=None,
                 config=None,
                 ):
        super().__init__(battle_format=config.battle_format,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self.except_cnt = 0
        self.total_cnt = 0
        self.save_replay_dir = save_replay_dir
        self.last_output = None
        self.last_state_prompt = None
        self.config = config

        # set tokenizer
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                                                    cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load llm as policy model
        model_kwargs = {'device_map': 'balanced'}
        policy_dtype = getattr(torch, config.model.policy_dtype)
        self.policy = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, output_hidden_states=True, cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
        disable_dropout(self.policy)

        if config.model.archive is not None:
            state_dict = torch.load(config.model.archive, map_location='cpu')
            step, metrics = state_dict['step_idx'], state_dict['metrics']
            print(
                f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
            self.policy.load_state_dict(state_dict['state'])

    def choose_move(self, battle: AbstractBattle):

        self.policy.eval()

        if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            next_action = BattleOrder(battle.available_switches[0])
            return next_action

        # state_prompt = self.state_translate(battle)
        system_prompt, state_prompt = self.state_translate(battle) # add lower case

        if battle.active_pokemon.fainted:
            constraint_prompt1 = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
        else:
            constraint_prompt1 = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''

        state_prompt1 = state_prompt + constraint_prompt1
        user_prompt = system_prompt + state_prompt1 + 'Output:{"'
        print("===================")
        print(user_prompt)

        input_dict = self.tokenizer(user_prompt, return_tensors="pt").to("cuda")

        next_action = None
        for i in range(5):
            try:
                with torch.no_grad():
                    outputs = self.policy.generate(
                                        inputs=input_dict["input_ids"],
                                        attention_mask=input_dict["attention_mask"],
                                        max_length=self.config.max_length,
                                        temperature=self.config.temperature,
                                        max_new_tokens=self.config.max_output_length,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.eos_token_id
                                    )
                llm_output = self.tokenizer.decode(outputs, skip_special_tokens=True)
                llm_output = llm_output.split("Output:")[1]
                next_action, _ = self.parse(llm_output, battle)
            except Exception as e:
                print(e)
                continue

        if next_action:
            print("LLM output:", llm_output)
            with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                f.write(json.dumps({"prompt": user_prompt, "llm_output": llm_output}) + "\n")
        else:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
            print("Exception occured.....")

        self.total_cnt += 1
        return next_action
