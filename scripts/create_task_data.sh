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

res="96x160"  # height x width
fps=8

ANNOS_DIR="../data"
TASKS_DIR="../data/tasks"
ORIGFPS_DIR="../data/videos/fps"
MODELRES_DIR="../data/videos/npy_${res}pix_${fps}fps"

for dset in uvo-valid uvo-test; do

    mkdir -p ${TASKS_DIR}/${dset}

    python create_task_data.py \
        ${ANNOS_DIR}/${dset}.json \
        ${ORIGFPS_DIR}/uvo_videos_dense,${ORIGFPS_DIR}/uvo_videos_sparse \
        ${MODELRES_DIR}/uvo_videos_dense,${MODELRES_DIR}/uvo_videos_sparse \
        ${TASKS_DIR}/${dset}

done

for dset in didemo-valid didemo-test oops-valid oops-test; do

    mkdir -p ${TASKS_DIR}/${dset}

    data=`echo ${dset} | cut -d '-' -f1`

    python create_task_data.py \
        ${ANNOS_DIR}/${dset}.json \
        ${ORIGFPS_DIR}/${data} \
        ${MODELRES_DIR}/${data} \
        ${TASKS_DIR}/${dset}

done
