// Copyright (c) MONAI Consortium
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

let websocket = null;

function getWebsocketProtocol() {
  return window.location.protocol === 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(port=49000) {
  return `${getWebsocketProtocol()}${window.location.hostname}:${port}`;
}

// Connects to the websocket server on a given port and sets up event handlers.
function connectWebsocket(port=49000, onMessageCallback=null) {
  const url = getWebsocketURL(port);
  console.log("Attempting WebSocket connection to:", url);
  websocket = new WebSocket(url);

  websocket.onopen = () => {
    console.log("WebSocket connected to", url);
  };

  websocket.onmessage = (event) => {
    console.log("Message received from server:", event.data);
    try {
      const msg = JSON.parse(event.data);
      // If we have a callback, call it
      if (onMessageCallback) onMessageCallback(msg);
    } catch (e) {
      console.error("Failed to parse WebSocket message as JSON:", e, event.data);
    }
  };

  websocket.onerror = (err) => {
    console.error("WebSocket error:", err);
  };

  websocket.onclose = () => {
    console.log("WebSocket closed");
  };
}

// Sends a JSON payload over the WebSocket
function sendJSON(payload) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) {
    console.warn("WebSocket not open. Unable to send message:", payload);
    return;
  }
  console.log("Sending JSON message:", payload);
  websocket.send(JSON.stringify(payload));
}