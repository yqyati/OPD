# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from verl.utils.tracking import _default_tensorboard_dir, _safe_path_component


def test_safe_path_component_leaves_short_names_unchanged():
    assert _safe_path_component("prefix512-opd") == "prefix512-opd"


def test_safe_path_component_shortens_long_utf8_names_stably():
    name = "实验-" * 200
    shortened = _safe_path_component(name)

    assert len(shortened.encode("utf-8")) <= 200
    assert shortened.endswith(_safe_path_component(name)[-14:])
    assert shortened == _safe_path_component(name)
    assert shortened != name


def test_tensorboard_default_uses_safe_components():
    path = _default_tensorboard_dir("project" * 100, "experiment" * 100)
    components = path.split("/")

    assert components[0] == "tensorboard_log"
    assert all(len(component.encode("utf-8")) <= 200 for component in components[1:])
