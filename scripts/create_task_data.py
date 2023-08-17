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
from typing import Any, List, Optional, Union

import numpy as np
import tqdm


# ============================================================================ #
#                                  PromptDict                                  #
# ============================================================================ #
PromptDict = dict[str, Any]
Prompt = Union[str, PromptDict]

def storybench_prompt_dict(
    # Texts to generate video with.
    texts: List[str],
    # How many steps to generate per text.
    durations: Optional[List[int]] = None,
    # Exact number of frames per text.
    frames_per_text: Optional[List[int]] = None,
    # Text describing background.
    background_text: Optional[str] = None,
    # Which video, if any, to condition on.
    npz_video: Optional[str] = None,
    # Which frame of npz_video to start from.
    npz_video_start_frame: int = 0,
    # Which frame of npz_video to end in. 
    # None takes the video until the end.
    npz_video_end_frame: Optional[int] = None,
    # WWhich frame of target ground-truth npz_video to start from.
    npz_gt_video_start_frame: Optional[int] = 0,
    # Which frame of target ground-truth npz_video to end in.
    # None takes the video until the end.
    npz_gt_video_end_frame: Optional[int] = None,
    # How many frames to skip in the result that is stored.
    skip_frames_after_generation: int = 0,
    # Or if the first generated video shold be skipped.
    storybench_mode: str = 'story_gen',
    # Comment to display in html page.
    comment: Optional[str] = None
    ) -> PromptDict:
    """Creates a Phenaki PromptDict."""

    assert storybench_mode in {'story_gen', 'story_cont', 'action_exe'}
    if storybench_mode == 'story_gen':
        assert background_text is not None

    return {'texts': texts,
            'durations': durations,
            'exact_frames_per_prompt': frames_per_text,
            'background': background_text,
            'npz_video': npz_video,
            'npz_video_start_frame': npz_video_start_frame,
            'npz_video_end_frame': npz_video_end_frame,
            'npz_gt_video_start_frame': npz_gt_video_start_frame,
            'npz_gt_video_end_frame': npz_gt_video_end_frame,
            'skip_frames_after_generation': skip_frames_after_generation,
            'storybench_mode': storybench_mode,
            'comment': comment}


# ============================================================================ #
#                                Task functions                                #
# ============================================================================ #
def create_storygen_data(annotations, npy_paths, video_to_fps):
    prompts = []

    for entry in tqdm.tqdm(annotations, total=len(annotations)):
        video_id = entry['video_name']
        fps = video_to_fps[video_id]
        for npy_path in npy_paths:
            video_fn = os.path.join(npy_path, f'{video_id}.npy')
            if os.path.exists(video_fn):
                with open(video_fn, 'rb') as f:
                    video_frames = np.load(f)

        background = entry['background_description']
        texts = [e.replace(' - ', '-').replace(' ,', ',')
                 for e in entry['sentence_parts']]

        # map original frames to target fps frames
        start_frames = [round(fps * s) for s in entry['start_times']]
        end_frames = [min(video_frames.shape[0], round(fps * s))
                      for s in entry['end_times']]
        # count number of frames per step
        exact_num_frames = [(e - s) for (s, e) in zip(start_frames, end_frames)]

        assert len(texts) == len(exact_num_frames)

        comment = entry['question_info']
        prompt = storybench_prompt_dict(
            texts=texts,
            background_text=background,
            durations=None,
            frames_per_text=exact_num_frames,
            npz_video=video_fn,
            npz_video_start_frame=0,
            npz_video_end_frame=0,
            skip_frames_after_generation=None,
            storybench_mode='story_gen',
            comment=comment,)
        prompts.append(prompt)
    
    return prompts


def create_storycont_data(annotations, npy_paths, video_to_fps, init_sec=0.5):
    prompts = []

    for entry in tqdm.tqdm(annotations, total=len(annotations)):
        video_id = entry['video_name']
        fps = video_to_fps[video_id]
        for npy_path in npy_paths:
            video_fn = os.path.join(npy_path, f'{video_id}.npy')
            if os.path.exists(video_fn):
                with open(video_fn, 'rb') as f:
                    video_frames = np.load(f)

        texts = [e.replace(' - ', '-').replace(' ,', ',')
                 for e in entry['sentence_parts']]

        # map original frames to target fps frames
        start_frames = [round(fps * s) for s in entry['start_times']]
        end_frames = [min(video_frames.shape[0], round(fps * s))
                      for s in entry['end_times']]
        # count number of frames per step
        exact_num_frames = [(e - s) for (s, e) in zip(start_frames, end_frames)]
        exact_num_frames[0] -= int(init_sec*fps)  # cont mode

        assert len(texts) == len(exact_num_frames)

        end_frame = start_frames[0] + int(init_sec*fps)  # cont mode
        comment = entry['question_info']
        prompt = storybench_prompt_dict(
            texts=texts,
            durations=None,
            frames_per_text=exact_num_frames,
            npz_video=video_fn,
            npz_video_start_frame=0,
            npz_video_end_frame=end_frame,
            skip_frames_after_generation=end_frame,
            storybench_mode='story_cont',
            comment=comment,)
        prompts.append(prompt)
    
    return prompts


def create_actionexe_data(annotations, npy_paths, video_to_fps, init_sec=0.5):
    prompts = []

    for entry in tqdm.tqdm(annotations, total=len(annotations)):
        video_id = entry['video_name']
        fps = video_to_fps[video_id]
        for npy_path in npy_paths:
            video_fn = os.path.join(npy_path, f'{video_id}.npy')
            if os.path.exists(video_fn):
                with open(video_fn, 'rb') as f:
                    video_frames = np.load(f)

        texts = [e.replace(' - ', '-').replace(' ,', ',')
                 for e in entry['sentence_parts']]

        # map original frames to target fps frames
        start_frames = [round(fps * s) for s in entry['start_times']]
        end_frames = [min(video_frames.shape[0], round(fps * s))
                      for s in entry['end_times']]
        # count number of frames per step
        exact_num_frames = [(e - s) for (s, e) in zip(start_frames, end_frames)]
        first_hist_len = max(int(init_sec*fps) - start_frames[0], 0)
        exact_num_frames[0] = exact_num_frames[0] - first_hist_len  # cont mode

        assert len(texts) == len(exact_num_frames)

        end_frame = max(start_frames[0], int(init_sec*fps))  # cont mode
        for text_ix, text in enumerate(texts):
            # target video end
            tgt_end_frm = end_frames[text_ix]
            tgt_end_frm = min(tgt_end_frm, len(video_frames))

            comment = entry['question_info'] + f'_{text_ix}'
        
            prompt = storybench_prompt_dict(
                texts=[text],
                durations=None,
                frames_per_text=[exact_num_frames[text_ix]],
                npz_video=video_fn,
                npz_video_start_frame=0,
                npz_video_end_frame=end_frame,
                skip_frames_after_generation=end_frame,
                npz_gt_video_start_frame=end_frame,
                npz_gt_video_end_frame=tgt_end_frm,
                storybench_mode='action_exe',
                comment=comment,)
            prompts.append(prompt)
            end_frame = tgt_end_frm  # update conditioning video end
    
    return prompts


# ============================================================================ #
#                                     MAIN                                     #
# ============================================================================ #
def main(annos_fn, fps_paths, npy_paths, out_path):
    video_to_fps = dict()
    for fps_path in fps_paths:
        with open(os.path.join(fps_path, 'video_to_fps.json')) as f:
            video_to_fps.update(json.load(f))

    with open(annos_fn) as f:
        annos = json.load(f)

    video_to_bg = dict()
    for e in annos:
        video_to_bg[e['video_name']] = e['background_description']

    # Story generation
    prompts = create_storygen_data(annos, npy_paths, video_to_fps)
    prompts_fn = os.path.join(out_path, f'story_gen.json')
    with open(prompts_fn, 'w') as f:
        json.dump(prompts, f)
    
    # Story continuation
    prompts = create_storycont_data(annos, npy_paths, video_to_fps)
    prompts_fn = os.path.join(out_path, f'story_cont.json')
    with open(prompts_fn, 'w') as f:
        json.dump(prompts, f)

    # Action execution
    prompts = create_actionexe_data(annos, npy_paths, video_to_fps)
    prompts_fn = os.path.join(out_path, f'action_exe.json')
    with open(prompts_fn, 'w') as f:
        json.dump(prompts, f)
    

if __name__ == "__main__":
    annos_fn = sys.argv[1]
    fps_paths = sys.argv[2].split(',')
    npy_paths = sys.argv[3].split(',')
    out_path = sys.argv[4]
    main(annos_fn, fps_paths, npy_paths, out_path)
