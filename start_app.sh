#!/bin/bash

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

LMM_LOG_FILE="${LMM_LOG_FILE:-./copilot_server.log}"

# Trap SIGINT and kill all processes in the group
trap 'kill 0' EXIT INT

# 1) Start the Whisper server (port 43001)
cd whisper/
python whisper_online_server.py \
    --model large-v3-turbo \
    --port 43001 \
    --warmup-file jfk.flac &
cd ..

sleep 3

# 2) Start vLLM server, which will serve models on port 8000 by default
bash run_vllm_server.sh >> "${LMM_LOG_FILE}" 2>&1 &

sleep 15

# 3) Start our Python app
python app.py &

wait
