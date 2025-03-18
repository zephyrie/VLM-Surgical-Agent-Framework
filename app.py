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

import asyncio
import logging
import os
from threading import Thread

from utils.chat_history import ChatHistory
from utils.response_handler import ResponseHandler

from agents.selector_agent import SelectorAgent
from agents.annotation_agent import AnnotationAgent
from agents.chat_agent import ChatAgent
from agents.notetaker_agent import NotetakerAgent
from agents.post_op_note_agent import PostOpNoteAgent

from web.webserver import Webserver

logging.basicConfig(level=logging.DEBUG)

async def main():
    chat_history = ChatHistory()
    response_handler = ResponseHandler()

    # Define the callback for messages coming from the WebSocket.
    def msg_callback(payload, msg_type, timestamp):
        """
        Called when the user manually types input or when the webserver passes along an ASR transcript.
        """
        if 'user_input' in payload:
            user_text = payload['user_input']
            logging.debug(f"Processing user input: {user_text}")
            try:
                # Let the selector decide which agent to pick
                selected_agent_name, corrected_text = selector_agent.process_request(
                    user_text, chat_history.to_list()
                )
                if not selected_agent_name:
                    logging.error("No agent selected by selector for user_input.")
                    return

                # If there's an image in the frame_queue, retrieve it
                frame_data = None
                if not web.frame_queue.empty():
                    try:
                        frame_data = web.frame_queue.get_nowait()
                    except Exception as e:
                        logging.error(f"Error retrieving frame data: {e}")

                # Pass the image (if any) along with empty tool labels
                visual_info = {"image_b64": frame_data, "tool_labels": {}}

                # If user input triggers PostOpNoteAgent, do final note generation
                if selected_agent_name == "PostOpNoteAgent":
                    # Stop the background annotation
                    annotation_agent.stop()

                    # Determine the procedure folder from annotation_agent
                    procedure_folder = os.path.dirname(annotation_agent.annotation_filepath)
                    
                    final_json = post_op_note_agent.generate_post_op_note(procedure_folder)
                    if final_json is None:
                        response_data = {
                            "name": "PostOpNoteAgent",
                            "response": "Failed to create final post-op note. Check logs."
                        }
                    else:
                        response_data = {
                            "name": "PostOpNoteAgent",
                            "response": "Final post-op note created. See post_op_note.json in the procedure folder."
                        }
                else:
                    agent = agents.get(selected_agent_name)
                    if agent:
                        response_data = agent.process_request(
                            corrected_text, chat_history.to_list(), visual_info
                        )
                    else:
                        response_data = {
                            "name": selected_agent_name,
                            "response": f"Agent '{selected_agent_name}' not implemented."
                        }

                # Update chat history
                chat_history.add_user_message(corrected_text)
                chat_history.add_bot_message(response_data["response"])

                # Send result to UI
                web.send_message({"agent_response": response_data["response"]})
            except Exception as e:
                logging.error(f"Error processing user_input: {e}", exc_info=True)

    # Create the webserver first so that its frame_queue is available.
    global web
    web = Webserver(web_server='0.0.0.0', web_port=8050, ws_port=49000, msg_callback=msg_callback)
    web.start()

    # Now create agents, passing web.frame_queue to the AnnotationAgent.
    selector_agent = SelectorAgent("configs/selector.yaml", response_handler)
    annotation_agent = AnnotationAgent("configs/annotation_agent.yaml", response_handler, frame_queue=web.frame_queue)
    chat_agent = ChatAgent("configs/chat_agent.yaml", response_handler)
    notetaker_agent = NotetakerAgent("configs/notetaker_agent.yaml", response_handler)
    post_op_note_agent = PostOpNoteAgent("configs/post_op_note_agent.yaml", response_handler)

    agents = {
        "ChatAgent": chat_agent,
        "NotetakerAgent": notetaker_agent,
        "PostOpNoteAgent": post_op_note_agent
    }

    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        logging.info("Shutting down gracefully.")

if __name__ == "__main__":
    asyncio.run(main())
