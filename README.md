# Surgical Copilot

Surgical Copilot is a multimodal agentic AI framework tailored for surgical procedures. It supports:

* **Speech-to-Text**: Real-time audio is captured, transcribed by Whisper.
* **VLM/LLM-based Conversational Agents**: A *selector agent* decides which specialized agent to invoke:
    *   ChatAgent for general Q&A,
    *   NotetakerAgent to record specific notes,
    *   AnnotationAgent to automatically annotate progress in the background,
    *   PostOpNoteAgent to summarize all data into a final post-operative note.
* **(Optional) Text-to-Speech**: The system can speak back the AI’s response if you enable TTS (ElevenLabs is implemented, but any local TTS could be implemented as well).
* **Computer Vision** or multimodal features are supported via a finetuned VLM (Vision Language Model), launched by Ollama.


## System Flow and Agent Overview

1. Microphone: The user clicks “Start Mic” in the web UI, or types a question.
2. Whisper ASR: Transcribes speech into text (via whisper_online_server.py).
3. SelectorAgent: Receives text from the UI, corrects it (if needed), decides whether to direct it to:
    * ChatAgent (general Q&A about the procedure)
    * NotetakerAgent (records a note with timestamp + optional image frame)
    * In the background, AnnotationAgent is also generating structured “annotations” every 10 seconds.
4. NotetakerAgent: If chosen, logs the note in a JSON file.
5. AnnotationAgent: Runs automatically, storing procedure annotations in ```procedure_..._annotations.json```.
6. PostOpNoteAgent (optional final step): Summarizes the entire procedure, reading from both the annotation JSON and the notetaker JSON, producing a final structured post-op note.

Installation 

1. Clone or Download this repository:

```
git clone https://github.com/project-monai/VLM-Surgical-Agent-Framework
cd VLM-Surgical-Agent-Framework
```

2. Install Dependencies:

```
conda create -n surgical_copilot python=3.12
conda activate surgical_copilot
pip install -r requirements.txt
```

3. Models Folder:

Download models from Huggingface here: TBD

* Place your model directory in ```models/```. The folder structure is: 

```
models/
  ├── Llama-3.2-11B-lora-surgical-4bit/
```

4. Video Setup: 

* Use the UI to select a surgical video sample to use.

5. Setup: 

* Edit ```start_app.sh``` if you need to change ports or model file names.

## Running Surgical Copilot

1. Run the script:

```
./start_app.sh
```

2. **Open** your browser at ```http://127.0.0.1:8050```. You should see the Surgical Copilot interface:
    * A video sample (```sample_video.mp4```)
    * Chat console
    * A "Start Mic" button to begin ASR.

3. Try speaking or Typing:
    * If you say “Take a note: The gallbladder is severely inflamed,” the system routes you to NotetakerAgent.
    * If you say “What are the next steps after dissecting the cystic duct?” it routes you to ChatAgent.

4. Background Annotations:
    * Meanwhile, ```AnnotationAgent``` writes a file like: ```procedure_2025_01_18__10_25_03_annotations.json``` in the annotations folder very 10 seconds with structured timeline data.

## Text-to-Speech (Optional)

If you want to enable TTS with ElevenLabs (or implement your own local TTS server):
    * Follow the instructions in the index.html or your code snippet that calls a TTS route or API.
    * Provide your TTS API key if needed.

## File Structure

A brief overview:

```
surgical_copilot/
├── agents
│   ├── annotation_agent.py
│   ├── base_agent.py
│   ├── chat_agent.py
│   ├── notetaker_agent.py
│   ├── post_op_note_agent.py
│   └── selector_agent.py
├── app.py
├── configs
│   ├── annotation_agent.yaml
│   ├── chat_agent.yaml
│   ├── notetaker_agent.yaml
│   ├── post_op_note_agent.yaml
│   └── selector.yaml
├── models
│   ├── mmproj-model-f16.gguf
│   └── surgical_copilot_Q_6.gguf
├── README.md        <-- this file
├── requirements.txt
├── start_app.sh     <-- main script to launch everything
├── whisper          <-- directory for whisper servers
│   ├── whisper_online_server.py
│   └── jfk.flac
└── web
    ├── static
    │   ├── audio.js
    │   ├── bootstrap.bundle.min.js
    │   ├── bootstrap.css
    │   ├── chat.css
    │   ├── favicon.ico
    │   ├── jquery-3.6.3.min.js
    │   ├── nvidia-logo.png
    │   ├── sample_video.mp4
    │   └── websocket.js
    ├── templates
    │   └── index.html
    └── webserver.py
```
