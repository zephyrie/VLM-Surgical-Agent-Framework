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

import threading
import time
import logging
import os
import json
import queue
from typing import List
from pydantic import BaseModel
from .base_agent import Agent

class SurgeryAnnotation(BaseModel):
    timestamp: str
    elapsed_time_seconds: float
    tools: List[str]
    anatomy: List[str]
    surgical_phase: str
    description: str

class AnnotationAgent(Agent):
    def __init__(self, settings_path, response_handler, frame_queue, agent_key=None, procedure_start_str=None):
        super().__init__(settings_path, response_handler, agent_key=agent_key)
        self._logger = logging.getLogger(__name__)
        self.frame_queue = frame_queue  
        self.time_step = self.agent_settings.get("time_step_seconds", 10)

        if procedure_start_str is None:
            procedure_start_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
        self.procedure_start_str = procedure_start_str
        self.procedure_start = time.time()


        base_output_dir = self.agent_settings.get("annotation_output_dir", "procedure_outputs")
        subfolder = os.path.join(base_output_dir, f"procedure_{self.procedure_start_str}")
        os.makedirs(subfolder, exist_ok=True)

        self.annotation_filepath = os.path.join(subfolder, "annotation.json")
        self._logger.info(f"AnnotationAgent writing annotations to: {self.annotation_filepath}")

        self.annotations = []
        self.stop_event = threading.Event()

        # Start the background loop in a separate thread.
        self.thread = threading.Thread(target=self._background_loop, daemon=True)
        self.thread.start()
        self._logger.info(f"AnnotationAgent background thread started (interval={self.time_step}s).")

    def _background_loop(self):
        while not self.stop_event.is_set():
            try:
                # Attempt to get image data from the frame queue.
                try:
                    frame_data = self.frame_queue.get_nowait()
                except queue.Empty:
                    self._logger.debug("No image data available; skipping annotation generation.")
                    time.sleep(self.time_step)
                    continue

                annotation = self._generate_annotation(frame_data)
                if annotation:
                    self.annotations.append(annotation)
                    self.append_json_to_file(annotation, self.annotation_filepath)
                    self._logger.debug(f"New annotation appended: {annotation}")
            except Exception as e:
                self._logger.error(f"Error generating annotation: {e}", exc_info=True)
            time.sleep(self.time_step)

    def _generate_annotation(self, frame_data):
        messages = []
        if self.agent_prompt:
            messages.append({"role": "system", "content": self.agent_prompt})
        user_content = "Please produce an annotation of the surgical scene based on the provided image, following the required schema."
        messages.append({"role": "user", "content": user_content})
        try:
            guided_params = {"guided_json": json.loads(self.grammar)}
            raw_json_str = self.stream_image_response(
                prompt=self.generate_prompt(user_content, []),
                image_b64=frame_data,
                temperature=0.3,
                extra_body=guided_params
            )
            self._logger.debug(f"Raw annotation response: {raw_json_str}")

            try:
                parsed = SurgeryAnnotation.model_validate_json(raw_json_str)
            except Exception as e:
                self._logger.warning(f"Annotation parse error: {e}")
                return None

            annotation_dict = parsed.dict()
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            annotation_dict["timestamp"] = timestamp_str
            annotation_dict["elapsed_time_seconds"] = time.time() - self.procedure_start

            return annotation_dict

        except Exception as e:
            self._logger.warning(f"Annotation generation error: {e}")
            return None

    def process_request(self, input_data, chat_history):
        return {
            "name": "AnnotationAgent",
            "response": "AnnotationAgent runs in the background and generates annotations only when image data is available."
        }

    def stop(self):
        self.stop_event.set()
        self._logger.info("Stopping AnnotationAgent background thread.")
        self.thread.join()
