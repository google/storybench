#!/bin/bash

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DATA_DIR="../data"
VIDLN_DIR="${DATA_DIR}/vidln"
LLM_DIR="${DATA_DIR}/llm_outputs"
ORIGFPS_DIR="${DATA_DIR}/videos/fps"
VIDEO_DIR="${DATA_DIR}/videos/raw"

python create_vidln_training_data.py \
    ${VIDLN_DIR}/UVO_dense_train.jsonl \
    ${LLM_DIR}/uvo_dense_train.json \
    ${VIDEO_DIR}/uvo_videos_dense \
    ${ORIGFPS_DIR}/uvo_videos_dense \
    ${DATA_DIR}/uvo_dense-train.pipeline.json
