# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import json
from typing import Literal
from pydantic import BaseModel
from .base_agent import Agent 

class SelectorOutput(BaseModel):
    corrected_input: str
    selection: Literal["ChatAgent", "NotetakerAgent", "PostOpNoteAgent"]

class SelectorAgent(Agent):
    def __init__(self, settings_path, response_handler):
        super().__init__(settings_path, response_handler)
        self._logger = logging.getLogger(__name__)

    def process_request(self, text, chat_history):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})

        user_text = (
            f"User said: {text}\n\n"
            "Please ONLY return JSON in the shape:\n"
            '{"corrected_input": "...", "selection": "ChatAgent"}\n'
            "with selection in [ChatAgent, NotetakerAgent, PostOpNoteAgent]."
        )
        messages.append({"role": "user", "content": user_text})

        self._logger.debug(f"SelectorAgent calling vLLM with user text: {text}")

        try:
            guided_params = {"guided_json": json.loads(self.grammar)}
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=self.ctx_length,
                extra_body=guided_params
            )
            raw_json_str = result.choices[0].message.content
            self._logger.debug(f"Raw JSON from vLLM: {raw_json_str}")
            raw_json_str = raw_json_str.replace("\\'", "'")

            parsed = SelectorOutput.model_validate_json(raw_json_str)
            selected_agent = parsed.selection
            corrected_text = parsed.corrected_input

            self._logger.debug(f"Selected agent: {selected_agent}, corrected text: {corrected_text}")
            return selected_agent, corrected_text

        except Exception as e:
            self._logger.error(f"Error in process_request: {e}", exc_info=True)
            return None, None
