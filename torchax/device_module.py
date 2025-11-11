# Copyright 2025 Google LLC
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

import torch
import mindspore as ms


def _is_in_bad_fork():
  return False


def manual_seed_all(seed):
  pass


def device_count():
  # 返回MindSpore可用设备数量
  return ms.context.get_context("device_id") + 1


def get_rng_state():
  return []


def set_rng_state(new_state, device):
  pass


def is_available():
  return True


def current_device():
  return 0


def get_amp_supported_dtype():
  # MindSpore支持的AMP数据类型
  return [torch.float16, torch.bfloat16, torch.float32]
