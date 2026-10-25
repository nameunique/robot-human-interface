from types import SimpleNamespace

import numpy as np

from robot_human_interface.gui.runtime import ReadinessReason
from robot_human_interface.gui.worker import PipelineWorker
from robot_human_interface.protocol import finalize_safe_command
from robot_human_interface.skeleton import RobotJointCommand


def test_unverified_physical_mapping_is_explained_in_gui(qtbot):
    worker = PipelineWorker()
    snapshot = SimpleNamespace(
        safe_command=finalize_safe_command(
            RobotJointCommand.humanoid(0.0, np.zeros(20), 1.0),
            free_base_active=True,
            balance_active=True,
            hardware_mapping_verified=False,
        )
    )
    facts = worker._snapshot_safety_facts(snapshot)
    assert not facts.valid
    assert facts.reason_code is ReadinessReason.HARDWARE_MAPPING_UNVERIFIED
    assert "аппаратном стенде" in facts.reason
