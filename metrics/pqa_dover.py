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

"""ENTRYPOINT for running Perceptual Quality Assessment with DOVER."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import numpy as np
import torch
from torch.utils import tensorboard

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import yaml

# pylint: disable=g-bad-import-order
from metrics import utils
from metrics.models.dover.datasets import ViewDecompositionDataset
from metrics.models.dover.models import DOVER


_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output', required=True)
_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data dir', required=True)

_DATASET = flags.DEFINE_string('dataset', 'uvo_valid', 'dataset_split name')
_MODEL = flags.DEFINE_string('model', 'sample', 'model name')
_TASK = flags.DEFINE_string('task', 'sample_task', 'story visualisation task')

_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'batch size')
_NUM_VIDEOS = flags.DEFINE_integer('num_videos', 1, 'num gen videos per gt vid')
_FAIL_ON_CPU = flags.DEFINE_boolean('fail_on_cpu', False, 'fail if run w/o GPU')


METRIC_NAME = 'pqa_dover'
MODEL_NAME = 'DOVER'


def compute_scores(dataset, model, device):
  """Compute Video Quality Assessment scores."""

  def _fuse_results(results: Sequence[float]):
    a, t = (results[0] - 0.1107) / 0.07355, (results[1] + 0.08285) / 0.03774
    x = a * 0.6104 + t * 0.3896
    return {
        'aesthetic': 1 / (1 + np.exp(-a)) * 100.,
        'technical': 1 / (1 + np.exp(-t)) * 100.,
        'overall': 1 / (1 + np.exp(-x)) * 100.,
    }

  all_results = []
  with logging_redirect_tqdm():
    for e in tqdm.tqdm(dataset, desc='Testing'):
      video = {}
      for key in ['aesthetic', 'technical']:
        if key in e:
          video[key] = e[key][None,:].to(device)
          b, c, t, h, w = video[key].shape
          video[key] = (
              video[key]
              .reshape(b, c, e['num_clips'][key], t//e['num_clips'][key], h, w)
              .permute(0, 2, 1, 3, 4, 5)
              .reshape(b * e['num_clips'][key], c, t//e['num_clips'][key], h, w)
          )

      with torch.no_grad():
        results = model(video, reduce_scores=False)
        results = [np.mean(l.cpu().numpy()) for l in results]
      rescaled_results = _fuse_results(results)

      rescaled_results.update({'name': e['name'].split('/')[-1].split('.')[0]})
      all_results.append(rescaled_results)
  return all_results


def main(_):
  logging.info('PQA job started')
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
  num_videos = _NUM_VIDEOS.value
  output_dir = _OUTPUT_DIR.value
  task_name = _TASK.value
  if output_dir:
    save_path = os.path.join(
        output_dir, model_name, task_name, dataset_name, METRIC_NAME
    )
    os.makedirs(save_path, exist_ok=True)
  write_metrics = output_dir is not None
  if write_metrics:
    writer = tensorboard.SummaryWriter(save_path)

  # Load model
  with open(
      f'{os.path.dirname(__file__)}/models/dover/{MODEL_NAME.lower()}.yml'
  ) as f:
    opt = yaml.safe_load(f)
  cache_dir = os.path.join(_DATA_DIR.value, 'checkpoints')
  opt['model']['args']['cache_dir'] = cache_dir
  cache_path = os.path.join(cache_dir, f'{MODEL_NAME}.pth')
  model = DOVER(**opt['model']['args']).to(device)
  model.load_state_dict(torch.load(cache_path, map_location=device))

  # Data
  preds_dir = os.path.join(
      _DATA_DIR.value, 'data', model_name, task_name, dataset_name)
  dopt = opt['data']['val-l1080p']['args']
  dopt['anno_file'] = None
  dopt['data_prefix'] = preds_dir

  # ========================================================================== #
  #                                    PQA                                     #
  # ========================================================================== #
  # Compute scores
  logging.info('Computing PQA.')
  outputs = []
  for vid_idx in range(num_videos):
    dataset = ViewDecompositionDataset(
        dopt, num_videos=num_videos, video_idx=vid_idx)
    out = compute_scores(dataset, model, device)
    outputs.append(out)

    with open(os.path.join(save_path, f'scores_{vid_idx}.tsv'), 'w') as f:
      f.write('name\taesthetic_score\ttechnical_score\toverall_score\n')
      for d in out:
        l = [str(d[k]) for k in ['name', 'aesthetic', 'technical', 'overall']]
        f.write('\t'.join(l) + '\n')

  metric_results = dict()
  for k in ['aesthetic', 'technical', 'overall']:
    results = [np.mean([d[k] for d in outputs[ix]]) for ix in range(num_videos)]
    metric_results[k] = results
    logging.info('PQA-%s: %s (%s ± %s)',
                 k, results, np.mean(results), np.std(results))

  if write_metrics:
    for k in metric_results:
      results = metric_results[k]
      for ix, res in enumerate(results):
        writer.add_scalar(f'PQA/{MODEL_NAME}({k})/{ix}', res)
      writer.add_scalar(f'PQA/{MODEL_NAME}({k})/mean', np.mean(results))
      writer.add_scalar(f'PQA/{MODEL_NAME}({k})/std', np.std(results))
      writer.flush()

      np.savetxt(os.path.join(save_path, f'result_{k}.txt'), results)

  if write_metrics:
    writer.close()

  logging.info('Job finished')


if __name__ == '__main__':
  app.run(main)
