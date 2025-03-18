"""
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
""" 

from abc import ABC, abstractmethod
import json
import logging
import yaml
import time
import tiktoken
from threading import Lock
import base64
import tempfile
import os
import requests
from openai import OpenAI

class Agent(ABC):
    _llm_lock = Lock()
    
    def __init__(self, settings_path, response_handler, agent_key=None):
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")        
        self.load_settings(settings_path, agent_key=agent_key)
        self.response_handler = response_handler
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.client = OpenAI(api_key="EMPTY", base_url=self.llm_url)
        self._wait_for_server()

    def load_settings(self, settings_path, agent_key=None):
        with open(settings_path, 'r') as f:
            full_config = yaml.safe_load(f)
        if agent_key and agent_key in full_config:
            self.agent_settings = full_config[agent_key]
        else:
            self.agent_settings = full_config
        self.description = self.agent_settings.get('description', '')
        self.max_prompt_tokens = self.agent_settings.get('max_prompt_tokens', 3000)
        self.ctx_length = self.agent_settings.get('ctx_length', 2048)
        self.agent_prompt = self.agent_settings.get('agent_prompt', '').strip()
        self.user_prefix = self.agent_settings.get('user_prefix', '')
        self.bot_prefix = self.agent_settings.get('bot_prefix', '')
        self.bot_rule_prefix = self.agent_settings.get('bot_rule_prefix', '')
        self.end_token = self.agent_settings.get('end_token', '')
        self.grammar = self.agent_settings.get('grammar', None)
        self.model_name = self.agent_settings.get('model_name', 'llama3.2')
        self.publish_settings = self.agent_settings.get('publish', {})
        self.llm_url = self.agent_settings.get('llm_url', "http://localhost:8000/v1")
        self.tools = self.agent_settings.get('tools', {})
        self._logger.debug(f"Agent config loaded. llm_url={self.llm_url}, model_name={self.model_name}")

    def _wait_for_server(self, timeout=30):
        attempts = 0
        check_url = f"{self.llm_url}/models"
        while attempts < timeout:
            try:
                r = requests.get(check_url)
                if r.status_code == 200:
                    self._logger.debug(f"Connected to vLLM server at {self.llm_url}")
                    return
            except Exception as e:
                self._logger.debug(f"Waiting for vLLM server (attempt {attempts+1}): {e}")
            time.sleep(1)
            attempts += 1
        raise ConnectionError(f"Unable to connect to vLLM server at {self.llm_url} after {timeout} seconds")

    def stream_response(self, prompt, grammar=None, temperature=0.0, display_output=True):
        with Agent._llm_lock:
            user_message = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
            request_messages = []
            if self.agent_prompt:
                request_messages.append({"role": "system", "content": self.agent_prompt})
            request_messages.append({"role": "user", "content": user_message})
            self._logger.debug(
                f"Sending chat request to vLLM/OpenAI client. Model={self.model_name}, temperature={temperature}\nUser message:\n{user_message[:500]}"
            )
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=request_messages,
                    temperature=temperature,
                    max_tokens=self.ctx_length
                )
                response_text = completion.choices[0].message.content if completion.choices else ""
                if display_output and self.response_handler:
                    self.response_handler.add_response(response_text)
                    self.response_handler.end_response()
                return response_text
            except Exception as e:
                self._logger.error(f"vLLM chat request failed: {e}", exc_info=True)
                return ""

    def stream_image_response(self, prompt, image_b64, grammar=None, temperature=0.0, display_output=True, extra_body=None):
        self._logger.debug(f"stream_image_response with model={self.model_name}")
        if not image_b64:
            raise ValueError("No image data provided for image response")
        user_message = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        try:
            raw_b64 = self._extract_raw_base64(image_b64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                file_path = tmp_file.name
                tmp_file.write(base64.b64decode(raw_b64))
            self._logger.debug(f"Temp image file created: {file_path}")
            messages = []
            if self.agent_prompt:
                messages.append({"role": "system", "content": self.agent_prompt})
            messages.append({
                "role": "user",
                "content": user_message,
                "images": [file_path]
            })
            request_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": self.ctx_length
            }
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body
            result = self.client.chat.completions.create(**request_kwargs)
            raw_text = result.choices[0].message.content
            if display_output and self.response_handler:
                self.response_handler.add_response(raw_text)
                self.response_handler.end_response()
            os.remove(file_path)
            return raw_text
        except Exception as e:
            self._logger.error(f"vLLM vision request failed: {e}", exc_info=True)
            raise

    def _extract_raw_base64(self, image_b64: str) -> str:
        prefix = "data:image/"
        if image_b64.startswith(prefix):
            parts = image_b64.split(',', 1)
            if len(parts) == 2:
                return parts[1]
            else:
                return image_b64
        else:
            return image_b64

    def generate_prompt(self, text, chat_history):
        system_prompt = f"{self.bot_rule_prefix}\n{self.agent_prompt}\n{self.end_token}"
        user_prompt = f"\n{self.user_prefix}\n{text}\n{self.end_token}"
        token_usage = self.calculate_token_usage(system_prompt + user_prompt)
        chat_prompt = self.create_conversation_str(chat_history, token_usage)
        prompt = system_prompt + chat_prompt + user_prompt
        prompt += f"\n{self.bot_prefix}\n"
        return prompt

    def create_conversation_str(self, chat_history, token_usage, conversation_length=2):
        total_tokens = token_usage
        msg_hist = []
        for user_msg, bot_msg in chat_history[:-1][-conversation_length:][::-1]:
            if bot_msg:
                bot_msg_str = f"\n{self.bot_prefix}\n{bot_msg}\n{self.end_token}"
                bot_tokens = self.calculate_token_usage(bot_msg_str)
                if total_tokens + bot_tokens > self.max_prompt_tokens:
                    break
                total_tokens += bot_tokens
                msg_hist.append(bot_msg_str)
            if user_msg:
                user_msg_str = f"\n{self.user_prefix}\n{user_msg}\n{self.end_token}"
                user_tokens = self.calculate_token_usage(user_msg_str)
                if total_tokens + user_tokens > self.max_prompt_tokens:
                    break
                total_tokens += user_tokens
                msg_hist.append(user_msg_str)
        return "".join(msg_hist[::-1])

    def calculate_token_usage(self, text):
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def process_request(self, input_data, chat_history):
        pass

    def append_json_to_file(self, json_object, file_path):
        try:
            if not os.path.isfile(file_path):
                with open(file_path, 'w') as f:
                    json.dump([json_object], f, indent=2)
            else:
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                if not isinstance(data, list):
                    data = []
                data.append(json_object)
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            self._logger.error(f"append_json_to_file error: {e}", exc_info=True)
