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

"""ENTRYPOINT for running FVD with InternVideo."""

import glob
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import torch
from torch.utils import tensorboard

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# pylint: disable=g-bad-import-order
from metrics.models import InternVideo
from metrics import utils


_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output', required=True)
_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data dir', required=True)

_DATASET = flags.DEFINE_string('dataset', 'uvo_valid', 'dataset_split name')
_MODEL = flags.DEFINE_string('model', 'sample', 'model name')
_TASK = flags.DEFINE_string('task', 'sample_task', 'story visualisation task')

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'batch size')
_NUM_VIDEOS = flags.DEFINE_integer('num_videos', 1, 'num gen videos per gt vid')
_FAIL_ON_CPU = flags.DEFINE_boolean('fail_on_cpu', False, 'fail if run w/o GPU')


METRIC_NAME = 'fvd_internvideo'
MODEL_NAME = 'InternVideo-MM-L-14'  # 768 feats
MODEL_DATA = 'MM'


def extract_features(model, data_dir, batch_size, device,
                     save_path=None, num_vids=1, vid_idx=0):
  """Extract video features."""
  del batch_size
  raw_dir = os.path.join(data_dir, 'raw')
  feats_dir = os.path.join(data_dir, 'features', METRIC_NAME)
  os.makedirs(feats_dir, exist_ok=True)

  out_fn = f'embeddings_{vid_idx}.npz'
  feats_fn = os.path.join(feats_dir, out_fn)
  save_fn = os.path.join(save_path, out_fn) if save_path else ''
  if os.path.exists(feats_fn):
    embeddings, names = utils.load_npz(feats_fn)
  elif os.path.exists(save_fn):
    embeddings, names = utils.load_npz(save_fn)
  else:
    embeddings, names = [], []
    fns = glob.glob(os.path.join(raw_dir, '*.npz'))
    with logging_redirect_tqdm():
      for fn in tqdm.tqdm(fns, total=len(fns)):
        frames_arr = np.load(fn, allow_pickle=True)['video'].clip(0, 255).astype(np.uint8)
        assert frames_arr.shape[-1] == 3  # [T, H, W*S, 3]
        # Generated videos are concat along width dim. Split along that axis
        video = np.split(frames_arr, num_vids, axis=2)[vid_idx]
        if video.shape[0] < 8:
          # duplicate
          repeats = [None, 8, 4, 3, 2, 2, 2, 2]
          video = video.repeat(repeats[video.shape[0]], 0)
        video = video[
            np.linspace(
                0, video.shape[0] - 1, video.shape[0] // 8 * 8, dtype=int
            )
        ]
        video = InternVideo.preprocess_video(video).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
          video_features = model.encode_video(video.unsqueeze(0))
          video_features = video_features.mean(0, keepdim=True)
          embeddings += video_features.cpu().tolist()

        names.append(fn.split('/')[-1].split('.')[0])
    embeddings = np.array(embeddings)

    if save_path:
      utils.save_npz(embeddings, names, save_fn)

  return embeddings, names


def main(_):
  logging.info('FVD job started')
  device = torch.device(utils.check_gpu(_FAIL_ON_CPU.value))
  logging.info('data_dir: %s', _DATA_DIR.value)
  logging.info('output_dir: %s', _OUTPUT_DIR.value)
  logging.info('dataset: %s', _DATASET.value)
  logging.info('model: %s', _MODEL.value)
  logging.info('task: %s', _TASK.value)
  logging.info('batch_size: %s', _BATCH_SIZE.value)
  logging.info('num_videos: %s', _NUM_VIDEOS.value)
  logging.info('fail_on_cpu: %s', _FAIL_ON_CPU.value)

  dataset_name = _DATASET.value
  model_name = _MODEL.value
  output_dir = _OUTPUT_DIR.value
  task_name = _TASK.value
  if output_dir:
    save_path = os.path.join(
        output_dir, model_name, task_name, dataset_name, METRIC_NAME)
    os.makedirs(save_path, exist_ok=True)
  write_metrics = output_dir is not None
  if write_metrics:
    writer = tensorboard.SummaryWriter(save_path)

  # Load model
  cache_dir = os.path.join(_DATA_DIR.value, 'checkpoints')
  model = InternVideo.load_model(f'{cache_dir}/{MODEL_NAME}.ckpt')
  model = model.eval().to(device)

  # Extract model prediction features
  logging.info('Extracting predictions features.')
  preds_dir = os.path.join(
      _DATA_DIR.value, 'data', model_name, task_name, dataset_name)
  preds_embs_list, preds_names_list = [], []
  for vid_idx in range(_NUM_VIDEOS.value):
    preds_embs, preds_names = extract_features(
        model, preds_dir, _BATCH_SIZE.value, device,
        save_path=save_path, num_vids=_NUM_VIDEOS.value, vid_idx=vid_idx)
    preds_embs_list.append(preds_embs)
    preds_names_list.append(preds_names)

  if model_name == 'ground_truth': exit()

  # Extract ground truth features
  logging.info('Extracting ground truth features.')
  truth_dir = os.path.join(
      _DATA_DIR.value, 'data', 'ground_truth', task_name, dataset_name)
  truth_embs, truth_names = extract_features(
      model, truth_dir, _BATCH_SIZE.value, device)

  # ========================================================================== #
  #                                    FVD                                     #
  # ========================================================================== #
  # Compute frechet distance
  logging.info('Computing FVD.')
  sort_ixs = np.argsort(truth_names)
  truth_embs = truth_embs[sort_ixs]
  truth_names = np.array(truth_names)[sort_ixs]
  results = []
  for preds_embs, preds_names in zip(preds_embs_list, preds_names_list):
    sort_ixs = np.argsort(preds_names)
    preds_embs = preds_embs[sort_ixs]
    preds_names = np.array(preds_names)[sort_ixs]
    assert list(preds_names) == list(truth_names)
    res = utils.get_frechet_dist(truth_embs, preds_embs)
    results.append(res)
  logging.info('FVD: %s (%s ± %s)', results, np.mean(results), np.std(results))

  if write_metrics:
    for ix, res in enumerate(results):
      writer.add_scalar(f'FVD/{MODEL_NAME}({MODEL_DATA})/{ix}', res)
    writer.add_scalar(f'FVD/{MODEL_NAME}({MODEL_DATA})/mean', np.mean(results))
    writer.add_scalar(f'FVD/{MODEL_NAME}({MODEL_DATA})/std', np.std(results))
    writer.flush()

  np.savetxt(os.path.join(save_path, 'result.txt'), results)

  if write_metrics:
    writer.close()

  logging.info('Job finished')


if __name__ == '__main__':
  app.run(main)
