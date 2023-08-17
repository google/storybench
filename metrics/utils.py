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

"""Utils functions for StoryBench evaluations."""

import io

from absl import logging

import numpy as np
from scipy import linalg
import torch


def check_gpu(fail_on_cpu: bool = False) -> str:
  """Print GPU info and return 'cuda' if found, 'cpu' otherwise."""
  try:
    logging.info('FLAGS.fail_on_cpu: %s', fail_on_cpu)
    logging.info('torch.__version__: %s', torch.__version__)
    logging.info('torch.cuda.device_count(): %s', torch.cuda.device_count())
    logging.info('torch.cuda.current_device(): %s', torch.cuda.current_device())
    logging.info(
        'torch.cuda.get_device_name(0): %s', torch.cuda.get_device_name(0)
    )
    logging.info('torch.cuda.is_available(0): %s', torch.cuda.is_available())
    if torch.cuda.is_available():
      return 'cuda'
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(e)
  if fail_on_cpu:
    logging.error('Not able to run on CPU')
    exit(1)
  logging.error('Falling back to CPU.')
  return 'cpu'


def save_npz(features: np.ndarray, names: list, path: str):
  """Save features and corresponding video names into an npz file."""
  with io.BytesIO() as io_bytes:
    np.savez_compressed(io_bytes, features=features, names=names)
    io_bytes.seek(0)
    data_to_write = io_bytes.read()
  with open(path, 'wb') as f:
    f.write(data_to_write)


def load_npz(path: str):
  """Load features and corresponding video names from an npz file."""
  with open(path, 'rb') as f:
    features = np.load(f, allow_pickle=True)['features']
    names = list(np.load(f, allow_pickle=True)['names'])
  return features, names


def get_frechet_dist(truth_feats: np.ndarray, preds_feats: np.ndarray) -> float:
  """Compute Frechet Distance between ground-truth and predicted features."""
  mu1 = np.mean(truth_feats, axis=0)
  sigma1 = np.cov(truth_feats, rowvar=False)
  mu2 = np.mean(preds_feats, axis=0)
  sigma2 = np.cov(preds_feats, rowvar=False)
  dist = frechet_distance(mu1, sigma1, mu2, sigma2)
  return dist


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                     mu2: np.ndarray, sigma2: np.ndarray,
                     eps: float = 1e-6):
  """Numpy implementation of the Frechet Distance.

  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1,C_1) and
  X_2 ~ N(mu_2,C_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

  Stable version by Dougal J. Sutherland.
  Implementation follows [https://github.com/eyalbetzalel/fcd/blob/main/fcd.py]

  Args:
      mu1: mean value for each feature dimension across all images by 1
      sigma1: covariance value for each feature dimension across all images by 1
      mu2: mean value for each feature dimension across all images by 2
      sigma2: covariance value for each feature dimension across all images by 2
      eps: small value to avoid singular product
  Returns:
      frechet_distance (float)
  """

  assert mu1.shape == mu2.shape, 'Truth and preds mean vectors have diff length'
  assert sigma1.shape == sigma2.shape, 'Truth and preds covars have diff dimens'

  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    logging.info('FID calculation produces singular product; '
                 'adding %s to diagonal of cov estimates', eps)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      raise ValueError(f'Imaginary component {np.max(np.abs(covmean.imag))}')
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
