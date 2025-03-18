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


let audioContext = null;
let mediaStream = null;
let recorder = null;
let audioChunks = [];
let audioWS = null;
let recording = false;

function getWebsocketProtocol() {
    return window.location.protocol === 'https:' ? 'wss://' : 'ws://';
}

async function startAudio() {
    if (recording) return;
    recording = true;

    try {
        // Request microphone
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

        // Use MediaRecorder to capture compressed WebM/Opus
        recorder = new MediaRecorder(mediaStream, { mimeType: "audio/webm" });
        audioChunks = [];

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        recorder.start();
        console.log("MediaRecorder started, mic live.");
    } catch (err) {
        console.error("Error starting audio:", err);
        recording = false;
        throw err;
    }
}

async function stopAudio() {
    if (!recording) return;
    recording = false;

    try {
        if (recorder && recorder.state !== "inactive") {
            recorder.stop();
            console.log("Recorder stopped.");
        }

        // Wait briefly for final data
        await new Promise(resolve => setTimeout(resolve, 100));

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        // Now send the entire .webm chunk via a one-shot WebSocket
        if (audioChunks.length > 0) {
            console.log("Total recorded chunks:", audioChunks.length);
            const fullBlob = new Blob(audioChunks, { type: "audio/webm" });
            const arrayBuf = await fullBlob.arrayBuffer();

            const audioUrl = `${getWebsocketProtocol()}${window.location.hostname}:49001`;
            audioWS = new WebSocket(audioUrl);
            audioWS.binaryType = "arraybuffer";

            audioWS.onopen = () => {
                console.log("Audio WS connected (one-shot). Sending .webm data...");
                audioWS.send(arrayBuf);
                audioWS.close();
            };

            audioWS.onclose = () => {
                console.log("One-shot audio WebSocket closed.");
            };
        } else {
            console.log("No audio chunks available to send.");
        }

        console.log("Stopped audio capture fully.");
    } catch (err) {
        console.error("Error stopping audio:", err);
    }
}
