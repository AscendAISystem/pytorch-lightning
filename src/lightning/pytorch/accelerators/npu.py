# Copyright The Lightning AI team.
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
import logging
import os
from typing import Any, Optional, Union

import torch
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator

_log = logging.getLogger(__name__)


class NPUAccelerator(Accelerator):
    """Accelerator for Huawei Ascend NPU devices."""

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not of type NPU.
            RuntimeError:
                If NPU is not available.
        """
        if device.type != "npu":
            raise ValueError(f"Device should be NPU, got {device} instead.")
        if not torch.npu.is_available():
            raise RuntimeError("NPU is not available.")
        torch.npu.set_device(device)

    @override
    def setup(self, trainer: "pl.Trainer") -> None:
        self.set_ascend_flags(trainer.local_rank)
        if torch.npu.is_available():
            torch.npu.empty_cache()

    @staticmethod
    def set_ascend_flags(local_rank: int) -> None:
        all_device_ids = ",".join(str(x) for x in range(NPUAccelerator.auto_device_count()))
        devices = os.getenv("ASCEND_RT_VISIBLE_DEVICES", all_device_ids)
        _log.info(f"LOCAL_RANK: {local_rank} - ASCEND_RT_VISIBLE_DEVICES: [{devices}]")

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Gets stats for the given NPU device.

        Args:
            device: NPU device for which to get stats

        Returns:
            A dictionary mapping the metrics to their values.
        """
        return torch.npu.memory_stats(device)

    @override
    def teardown(self) -> None:
        if torch.npu.is_available():
            torch.npu.empty_cache()

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Optional[list[int]]:
        """Accelerator device parsing logic for NPU devices."""
        if devices is None:
            return None
        if isinstance(devices, int) and devices == 0:
            return None
        if isinstance(devices, str) and str(devices).strip() in ("0", "[]"):
            return None

        # Normalize string input
        if isinstance(devices, str):
            if devices == "-1":
                return list(range(NPUAccelerator.auto_device_count()))
            if "," in devices:
                parsed = [int(x.strip()) for x in devices.split(",") if len(x.strip()) > 0]
            else:
                parsed = [int(devices.strip())]
        elif isinstance(devices, int):
            if devices == -1:
                return list(range(NPUAccelerator.auto_device_count()))
            parsed = list(range(devices))
        elif isinstance(devices, (list, tuple)):
            parsed = list(devices)
        else:
            return None

        # Validate that the requested devices are available
        available = NPUAccelerator.auto_device_count()
        for gpu_id in parsed:
            if gpu_id < 0 or gpu_id >= available:
                raise ValueError(
                    f"You requested NPU device {gpu_id} but this machine has only {available} NPU devices."
                )

        return parsed if len(parsed) > 0 else None

    @staticmethod
    @override
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return [torch.device("npu", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.npu.device_count()
        return 0

    @staticmethod
    @override
    def is_available() -> bool:
        return hasattr(torch, "npu") and torch.npu.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "npu"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
