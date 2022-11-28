## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines Tensorflow models."""

from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions

MINILM_L12_H384_UNCASED_INT32_SEQLEN128 = common_definitions.Model(
    id=unique_ids.MODEL_MINILM_L12_H384_UNCASED_INT32_SEQLEN128,
    name="MiniLML12H384Uncased",
    tags=["int32", "seqlen128"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF,
    # Converted from https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/commit/44acabbec0ef496f6dbc93adadea57f376b7c0ec
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-seqlen128-tf-model.tar.gz",
    entry_function="predict",
    input_types=["1x128xi32", "1x128xi32", "1x128xi32"])

BERT_FOR_MASKED_LM_FP32_SEQLEN512 = common_definitions.Model(
    id=unique_ids.MODEL_BERT_FOR_MASKED_LM_FP32_SEQLEN512_TF,
    name="BertForMaskedLMTF",
    tags=["fp32", "seqlen512", "tensorflow"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF,
    # Converted from https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#tfbertformaskedlm
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/bert-for-masked-lm-seq512-tf-model.tar.gz",
    entry_function="forward",
    input_types=["1x512xi32", "1x512xi32"])