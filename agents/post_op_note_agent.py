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

import os
import json
import math
import logging
from .base_agent import Agent

class PostOpNoteAgent(Agent):
    def __init__(self, settings_path, response_handler=None, agent_key=None):
        super().__init__(settings_path, response_handler, agent_key=agent_key)
        self._logger = logging.getLogger(__name__)
        self.chunk_size = self.agent_settings.get("chunk_size", 20)

        self.schema_dict = {}
        if self.grammar:
            try:
                self.schema_dict = json.loads(self.grammar)
                self._logger.debug(f"Parsed grammar JSON schema: {self.schema_dict}")
            except json.JSONDecodeError as e:
                self._logger.error(f"Failed to parse 'grammar' as JSON: {e}")

    def process_request(self, input_data, chat_history, visual_info=None):
        return {
            "name": "PostOpNoteAgent",
            "response": "Invoke generate_post_op_note(procedure_folder) to produce and save the final note."
        }

    def generate_post_op_note(self, procedure_folder):
        annotation_json = os.path.join(procedure_folder, "annotation.json")
        notetaker_json = os.path.join(procedure_folder, "notetaker_notes.json")

        ann_list = self._load_json_array(annotation_json)
        ann_summary = self._chunk_summarize_annotation(ann_list)

        note_list = self._load_json_array(notetaker_json)
        notes_summary = self._chunk_summarize_notetaker(note_list)

        user_msg = (
            f"Annotated summary:\n{ann_summary}\n\n"
            f"Notetaker summary:\n{notes_summary}\n\n"
            "Now produce a final post-op note in JSON format that conforms to the grammar."
        )

        final_prompt = (
            f"{self.agent_prompt}\n"
            f"<|im_start|>user\n"
            f"{user_msg}\n"
            f"<|im_end|>\n"
            f"{self.bot_prefix}\n"
        )
        self._logger.debug(f"Final post-op prompt: {final_prompt[:500]}...")

        raw_resp = self._ask_for_json(final_prompt)
        self._logger.debug(f"PostOp raw response: {raw_resp[:500]}")

        try:
            final_json = json.loads(raw_resp)
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse final post-op note JSON: {e}\nRaw={raw_resp}")
            return None

        post_op_file = os.path.join(procedure_folder, "post_op_note.json")
        self._save_post_op_note(final_json, post_op_file)

        return final_json

    def _ask_for_json(self, prompt_text: str):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})

        user_content = prompt_text.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        messages.append({"role": "user", "content": user_content})

        self._logger.debug("Calling vLLM for JSON response.")
        result = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=self.ctx_length
        )
        return result.choices[0].message.content

    def _chunk_summarize_annotation(self, ann_list):
        if not ann_list:
            return "No annotation data found."
        
        lines = []
        for ann in ann_list:
            ts = ann.get("timestamp", "???")
            phase = ann.get("surgical_phase", "?")
            desc = ann.get("description", "?")
            lines.append(f"[{ts}] Phase={phase}, {desc}")

        return self._multi_step_chunk_summarize(lines, label="Annotation data")

    def _chunk_summarize_notetaker(self, note_list):
        if not note_list:
            return "No notetaker data found."

        lines = []
        for note in note_list:
            ts = note.get("timestamp", "???")
            txt = note.get("text", "(no text)")
            lines.append(f"[{ts}] {txt}")

        return self._multi_step_chunk_summarize(lines, label="Notetaker data")

    def _multi_step_chunk_summarize(self, lines, label="Data"):
        if len(lines) <= self.chunk_size:
            block = "\n".join(lines)
            return self._ask_for_summary(block, label)
        else:
            chunk_summaries = []
            total = len(lines)
            n_chunks = math.ceil(total / self.chunk_size)
            idx = 0
            for i in range(n_chunks):
                chunk = lines[idx:idx+self.chunk_size]
                idx += self.chunk_size
                chunk_text = "\n".join(chunk)
                sub_summary = self._ask_for_summary(chunk_text, f"{label} chunk {i+1}/{n_chunks}")
                chunk_summaries.append(sub_summary)

            final_block = "\n\n".join(chunk_summaries)
            final_summary = self._ask_for_summary(final_block, f"{label} final summary")
            return final_summary

    def _ask_for_summary(self, text_block, label="Data"):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})

        user_prompt = (
            f"You are summarizing {label}.\n"
            f"Here is the data:\n\n{text_block}\n\n"
            "Please produce a concise summary.\n"
        )
        messages.append({"role": "user", "content": user_prompt})

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=self.ctx_length
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            self._logger.error(f"Error summarizing {label} with vLLM: {e}")
            return ""

    def _load_json_array(self, filepath):
        if not os.path.isfile(filepath):
            self._logger.warning(f"File not found: {filepath}")
            return []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    self._logger.warning(f"{filepath} is not a JSON list.")
                    return []
                return data
        except Exception as e:
            self._logger.error(f"Error reading {filepath}: {e}", exc_info=True)
            return []

    def _save_post_op_note(self, note_json, filepath):
        try:
            with open(filepath, "w") as f:
                json.dump(note_json, f, indent=2)
            self._logger.info(f"Post-op note saved to: {filepath}")
        except Exception as e:
            self._logger.error(f"Error writing post-op note to {filepath}: {e}", exc_info=True)
