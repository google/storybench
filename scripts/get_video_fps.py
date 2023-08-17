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

import json
import os.path
import sys

import cv2
import glob
import tqdm


def get_fps(fn):
    name = os.path.splitext(os.path.basename(fn))[0]
    cap = cv2.VideoCapture(fn)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return name, fps


def main(fps_path, video_path):
    fps_dict = dict()
    fns = glob.glob(f'{video_path}/*')
    for fn in tqdm.tqdm(fns, total=len(fns)):
        name, fps = get_fps(fn)
        fps_dict[name] = fps
    with open(os.path.join(fps_path, 'video_to_fps.json'), 'w') as f:
        json.dump(fps_dict, f)


if __name__ == "__main__":
    fps_path = sys.argv[1]
    video_path = sys.argv[2]
    main(fps_path, video_path)
