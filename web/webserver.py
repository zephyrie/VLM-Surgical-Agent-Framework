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

import base64
import flask
import json
import time
import socket
import threading
import queue
import websockets
import logging
import requests

from websockets.sync.server import serve as websocket_serve
from flask import request, jsonify

class Webserver(threading.Thread):
    def __init__(self, web_server='0.0.0.0', web_port=8050, ws_port=49000,
                 audio_ws_port=49001, msg_callback=None):
        super().__init__(daemon=True)
        self.host = web_server
        self.port = web_port
        self.msg_callback = msg_callback
        self.audio_ws_port = audio_ws_port
        self.frame_queue = queue.Queue()

        self.app = flask.Flask(__name__)
        self.app.add_url_rule('/', view_func=self.on_index, methods=['GET'])
        self.app.add_url_rule('/api/tts', view_func=self.tts_route, methods=['POST'])

        # For text messages from WebSocket
        self.ws_queue = queue.Queue()
        self.ws_server = websocket_serve(self.on_websocket, host=self.host, port=ws_port)
        self.ws_thread = threading.Thread(target=lambda: self.ws_server.serve_forever(), daemon=True)

        # For single-chunk audio
        self.audio_ws_server = websocket_serve(self.on_audio_websocket, host=self.host, port=self.audio_ws_port)
        self.audio_ws_thread = threading.Thread(target=lambda: self.audio_ws_server.serve_forever(), daemon=True)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

        # Create a socket to talk to Whisper. We'll reconnect for each new audio session.
        self.create_whisper_socket()

    def create_whisper_socket(self):
        """Create a fresh socket to the Whisper server (port 43001)."""
        self.whisper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.whisper_socket.connect(('localhost', 43001))
        self._logger.debug("Created + connected fresh socket to Whisper server.")

    @staticmethod
    def on_index():
        return flask.render_template('index.html')

    def on_websocket(self, websocket):
        listener_thread = threading.Thread(target=self.websocket_listener, args=[websocket], daemon=True)
        listener_thread.start()
        # Send queued messages
        while True:
            msg = self.ws_queue.get()
            self._logger.debug(f"Sending message to client: {msg}")
            websocket.send(msg)

    def websocket_listener(self, websocket):
        while True:
            msg = websocket.recv()
            self._logger.debug(f"Received message from client (len={len(msg)}).")
            try:
                data = json.loads(msg)
                # If auto_frame flag is present, push frame_data into the frame_queue
                if data.get('auto_frame') == True:
                    frame_data = data.get('frame_data')
                    if frame_data:
                        self._logger.debug("Got auto_frame data from client.")
                        self.frame_queue.put(frame_data)
                    continue
                # Also check for 'frame_data' in non-auto messages
                frame_data = data.pop('frame_data', None)
                if frame_data:
                    self._logger.debug("Got frame_data from client.")
                    self.frame_queue.put(frame_data)
                if 'user_input' in data and self.msg_callback:
                    self._logger.debug(f"Sending user_input to msg_callback: {data}")
                    self.msg_callback(data, 0, int(time.time() * 1000))
            except json.JSONDecodeError:
                self._logger.warning("Invalid JSON from client.")
                continue

    def on_audio_websocket(self, websocket):
        self._logger.info("Audio websocket connected (one-shot).")
        audio_data = bytearray()
        try:
            while True:
                chunk = websocket.recv()
                if isinstance(chunk, bytes):
                    audio_data.extend(chunk)
                else:
                    break
        except websockets.exceptions.ConnectionClosed:
            self._logger.info("Audio websocket connection closed by client.")
        if len(audio_data) > 0:
            self._logger.debug(f"Forwarding final chunk of size {len(audio_data)} bytes to whisper server")
            self.whisper_socket.sendall(audio_data)
            self._logger.debug("Shutting down write side so Whisper sees EOF")
            self.whisper_socket.shutdown(socket.SHUT_WR)
            recognized_text = self.read_whisper_result()
            self._logger.debug(f"Got recognized_text from whisper: {recognized_text}")
            if recognized_text.strip():
                self._logger.debug("Requesting a frame from browser for final transcript.")
                self.send_message({
                    "request_frame": True,
                    "recognized_text": recognized_text
                })
            else:
                self._logger.debug("No recognized text found from whisper.")
            self._logger.debug("Closing whisper socket entirely. Re-initializing for next time.")
            self.whisper_socket.close()
            self.create_whisper_socket()
        else:
            self._logger.debug("No audio data received from client.")

    def read_whisper_result(self):
        result_buffer = b""
        while True:
            chunk = self.whisper_socket.recv(1024)
            if not chunk:
                break
            result_buffer += chunk
        lines = result_buffer.decode('utf-8', errors='replace').split('\n')
        recognized_text = ""
        for line in lines:
            line = line.strip()
            if line.startswith("0 0 "):
                recognized_text = line[4:]
        return recognized_text
    
    def tts_route(self):
        data = request.json
        text = data.get('text', '').strip()
        api_key = data.get('api_key', None)
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if not api_key:
            return jsonify({"error": "No API key provided"}), 400
        voice_id = "Gsx7OZuV8m2a7U7c5Ftw"
        model_id = "eleven_multilingual_v2"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,        
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code != 200:
                return jsonify({"error": f"ElevenLabs API error {r.status_code}: {r.text[:500]}"}), 400
            audio_base64 = base64.b64encode(r.content).decode('utf-8')
            return jsonify({"tts_base64": audio_base64})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def send_message(self, payload):
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        self._logger.debug(f"Queueing message for client: {payload}")
        self.ws_queue.put(payload)

    def run(self):
        self.ws_thread.start()
        self.audio_ws_thread.start()
        self._logger.info(f"Starting Flask app on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
