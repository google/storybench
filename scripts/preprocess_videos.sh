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

RAW_DIR="../data/videos/raw"
ORIGFPS_DIR="../data/videos/fps"
ORIGRES_DIR="../data/videos/npy_origpix_${fps}fps"
MODELRES_DIR="../data/videos/npy_${res}pix_${fps}fps"

width=`echo ${res} | cut -d 'x' -f2`
height=`echo ${res} | cut -d 'x' -f1`

for dset in uvo_videos_dense uvo_videos_sparse oops didemo; do

    mkdir -p ${ORIGRES_DIR}/${dset} ${MODELRES_DIR}/${dset} ${ORIGFPS_DIR}/${dset}

    # extract original video fps
    python get_video_fps.py ${ORIGFPS_DIR}/${dset} ${RAW_DIR}/${dset}
    
    for fn in ${RAW_DIR}/${dset}/*; do
        name=`basename ${fn%.*}`

        # ffmpeg preprocess at full resolution (used for evaluation)
        (
            ffmpeg -i $fn \
            -vf fps=${fps},scale=force_original_aspect_ratio=increase \
            -sws_flags lanczos+full_chroma_int+full_chroma_inp+accurate_rnd \
            -vcodec mjpeg -qmin 1 -q:v 2 \
            ${ORIGRES_DIR}/${dset}/${name}.mp4
        ) > /dev/null 2>&1

        # ffmpeg preprocess at model resolution (used for prompting)
        (
            ffmpeg -i $fn \
            -vf fps=${fps},scale=${width}:${height}:force_original_aspect_ratio=increase,crop=${width}:${height} \
            -sws_flags lanczos+full_chroma_int+full_chroma_inp+accurate_rnd \
            -vcodec mjpeg -qmin 1 -q:v 2 \
            ${MODELRES_DIR}/${dset}/${name}.mp4
        ) > /dev/null 2>&1

    done

    # convert mp4 videos to ndarray
    python mp4_to_npy.py ${ORIGRES_DIR}/${dset} True
    python mp4_to_npy.py ${MODELRES_DIR}/${dset} True

done
