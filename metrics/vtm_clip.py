
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

"""ENTRYPOINT for running VTM with open_clip."""

import glob
import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import torch
from torch.utils import tensorboard

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# pylint: disable=g-bad-import-order
import open_clip

from metrics import utils


_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output dir')
_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data directory')

_DATASET = flags.DEFINE_string('dataset', 'uvo_valid', 'dataset_split name')
_MODEL = flags.DEFINE_string('model', 'sample', 'model name')
_TASK = flags.DEFINE_string('task', 'sample_task', 'story visualisation task')

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'batch size')
_NUM_VIDEOS = flags.DEFINE_integer('num_videos', 1, 'num gen videos per gt vid')
_FAIL_ON_CPU = flags.DEFINE_boolean('fail_on_cpu', False, 'fail if run w/o GPU')

METRIC_NAME = 'vtm_clip'
MODEL_NAME = 'ViT-L-14-336'  # 768 feats
MODEL_DATA = 'openai'


def extract_features(model, preprocess, tokenizer, data_dir, batch_size, device,
                     save_path=None, num_vids=1, vid_idx=0):
  """Extract image and text features."""
  raw_dir = os.path.join(data_dir, 'raw')
  feats_dir = os.path.join(data_dir, 'features', METRIC_NAME)
  os.makedirs(feats_dir, exist_ok=True)

  t_out_fn = 't_embeddings.npz'
  t_feats_fn = os.path.join(feats_dir, t_out_fn)
  t_save_fn = os.path.join(save_path, t_out_fn) if save_path else ''
  if os.path.exists(t_feats_fn):
    t_embeddings, _ = utils.load_npz(t_feats_fn)
  elif os.path.exists(t_save_fn):
    t_embeddings, _ = utils.load_npz(t_save_fn)
  else:
    t_embeddings = []
  t_embeds_exist = bool(len(t_embeddings))
  v_out_fn = f'v_embeddings_{vid_idx}.npz'
  v_feats_fn = os.path.join(feats_dir, v_out_fn)
  v_save_fn = os.path.join(save_path, v_out_fn) if save_path else ''
  if os.path.exists(v_feats_fn):
    v_embeddings, names = utils.load_npz(v_feats_fn)
  elif os.path.exists(v_save_fn):
    v_embeddings, names = utils.load_npz(v_save_fn)
  else:
    v_embeddings, names = [], []
    fns = glob.glob(os.path.join(raw_dir, '*.npz'))
    with logging_redirect_tqdm():
      for fn in tqdm.tqdm(fns, total=len(fns)):
        frames_arr = np.load(fn, allow_pickle=True)['video'].clip(0, 255).astype(np.uint8)
        num_frames = np.load(fn, allow_pickle=True)['frames']
        assert frames_arr.shape[-1] == 3  # [T, H, W*S, 3]
        # Generated videos are concat along width dim. Split along that axis
        frames_arr = np.split(frames_arr, num_vids, axis=2)[vid_idx]
        v_embs = []
        for start_ix in range(0, len(frames_arr), batch_size):
          images = frames_arr[start_ix: start_ix + batch_size]
          images = [Image.fromarray(im_arr) for im_arr in images]
          images = torch.stack([preprocess(image) for image in images])
          images = images.to(device)
          with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            v_embs += image_features.cpu().tolist()
        name = fn.split('/')[-1].split('.')[0]
        for ix in range(len(num_frames)):
          start_ix = sum(num_frames[:ix])
          end_ix = sum(num_frames[:ix+1])
          names.append(f'{name}_{ix}')
          v_embeddings.append(np.mean(v_embs[start_ix:end_ix], axis=0))

        if not t_embeds_exist:
          texts = np.load(fn, allow_pickle=True)['texts'].tolist()
          assert len(texts) == len(num_frames)
          texts = tokenizer(texts).to(device)
          with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(texts)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            t_embeddings += text_features.cpu().tolist()

    v_embeddings = np.array(v_embeddings)
    t_embeddings = np.array(t_embeddings)

    if save_path:
      utils.save_npz(v_embeddings, names, v_save_fn)
      if not t_embeds_exist:
        utils.save_npz(t_embeddings, names, t_save_fn)

  assert v_embeddings.shape == t_embeddings.shape
  return v_embeddings, t_embeddings, names


def score(v_embs, t_embs):
  return (v_embs * t_embs).sum(axis=-1).mean() * 100.


def main(_):
  logging.info('VTM job started')
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
  model, _, preprocess = open_clip.create_model_and_transforms(
      MODEL_NAME, pretrained=MODEL_DATA, device=device, cache_dir=cache_dir)
  tokenizer = open_clip.get_tokenizer(MODEL_NAME)

  # Extract model prediction features
  logging.info('Extracting features.')
  preds_dir = os.path.join(
      _DATA_DIR.value, 'data', model_name, task_name, dataset_name)
  v_embs_list, t_embs_list = [], []
  for vid_idx in range(_NUM_VIDEOS.value):
    v_embs, t_embs, _ = extract_features(
        model, preprocess, tokenizer, preds_dir, _BATCH_SIZE.value, device,
        save_path=save_path, num_vids=_NUM_VIDEOS.value, vid_idx=vid_idx)
    v_embs_list.append(v_embs)
    t_embs_list.append(t_embs)

  # ========================================================================== #
  #                                    VTM                                     #
  # ========================================================================== #
  # Compute video--text matching
  logging.info('Computing VTM.')
  results = []
  for v_embs, t_embs in zip(v_embs_list, t_embs_list):
    res = score(v_embs, t_embs)
    results.append(res)
  logging.info('VTM: %s (%s ± %s)', results, np.mean(results), np.std(results))
  if write_metrics:
    for ix, res in enumerate(results):
      writer.add_scalar(f'VTM/{MODEL_NAME}({MODEL_DATA})/{ix}', res)
    writer.add_scalar(f'VTM/{MODEL_NAME}({MODEL_DATA})/mean', np.mean(results))
    writer.add_scalar(f'VTM/{MODEL_NAME}({MODEL_DATA})/std', np.std(results))
    writer.flush()

  np.savetxt(os.path.join(save_path, 'result.txt'), results)

  if write_metrics:
    writer.close()

  logging.info('Job finished')


if __name__ == '__main__':
  app.run(main)
