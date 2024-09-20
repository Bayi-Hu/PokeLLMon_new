from src.player.gpt_player import GPTPlayer
from src.player.player import BattleOrder
from src.environment.abstract_battle import AbstractBattle
from src.utils.llm_utils import disable_dropout, get_local_dir
import json
import transformers
import torch

class LLMPlayer(GPTPlayer):
    def __init__(self,
                 model,
                 config,
                 save_replay_dir,
                 account_configuration=None,
                 server_configuration=None,
                 ):
        super().__init__(config=config,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self.model = model
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

    def choose_move(self, battle: AbstractBattle):

        self.model.eval()

        if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            next_action = BattleOrder(battle.available_switches[0])
            return next_action

        # state_prompt = self.state_translate(battle)
        system_prompt, state_prompt = self.state_translate(battle)

        if battle.active_pokemon.fainted:
            constraint_prompt = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
        else:
            constraint_prompt = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''

        state_prompt = system_prompt + state_prompt + constraint_prompt + 'Output:{"'
        # print("===================")
        print(state_prompt)

        input_dict = self.tokenizer(state_prompt, return_tensors="pt").to("cuda")

        next_action = None
        for i in range(5):
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                                        inputs=input_dict["input_ids"],
                                        attention_mask=input_dict["attention_mask"],
                                        # max_length=self.config.max_length,
                                        temperature=self.config.temperature,
                                        max_new_tokens=self.config.max_output_length,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        pad_token_id=self.tokenizer.eos_token_id
                                    )
                # print(outputs)
                llm_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                llm_output = llm_output.split("Output:")[1]
                next_action = self.parse(llm_output, battle)
                if next_action:
                    break
            except Exception as e:
                # print(str(e) + ":" +llm_output)
                continue

        if next_action:
            print("LLM output:", llm_output)
            with open(f"{self.save_replay_dir}/output.jsonl", "a") as f:
                f.write(json.dumps({"prompt": state_prompt, "llm_output": llm_output}) + "\n")
        else:
            self.except_cnt += 1
            next_action = self.choose_max_damage_move(battle)
            print("Exception occured.....")

        self.total_cnt += 1
        return next_action
