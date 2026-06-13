"""Quiet the noisy CUDA / TF plugin-registration warnings on Kaggle.


These are harmless — they happen because Kaggle's base image has both

TensorFlow and PyTorch preinstalled, and they race to register the same

cuFFT / cuDNN / cuBLAS plugin factories. Suppressing them keeps the

Kaggle cell output readable and stops them being mistaken for errors.

"""

from __future__ import annotations

import os
import warnings


def quiet_cuda_warnings() -> None:

    # TensorFlow is installed on Kaggle by default; hide its info logs.

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # HuggingFace + transformers advisory spam

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.environ.setdefault("WANDB_SILENT", "false")

    warnings.filterwarnings("ignore", category=UserWarning)

    warnings.filterwarnings("ignore", category=FutureWarning)
