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

import os.path
import sys

import cv2
import glob
import numpy as np
import tqdm


def save_video_as_array(fn, video_path, delete_video=False):
    name = os.path.splitext(os.path.basename(fn))[0]
    cap = cv2.VideoCapture(fn)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in range(frame_count):
        _, img = cap.read()
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (T, H, W, C)

    with open(os.path.join(video_path, f'{name}.npy'), 'wb') as f:
        np.save(f, frames)
    
    if delete_video:
        os.remove(fn)


def main(video_path, delete_videos):
    fns = glob.glob(f'{video_path}/*.mp4')
    for fn in tqdm.tqdm(fns, total=len(fns)):
        save_video_as_array(fn, video_path, delete_videos)


if __name__ == "__main__":
    video_path = sys.argv[1]
    delete_videos = eval(sys.argv[2].capitalize())
    main(video_path, delete_videos)
