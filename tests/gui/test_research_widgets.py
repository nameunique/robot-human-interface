from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QSignalSpy

from robot_human_interface.experiments import ExperimentSpec, RecorderState
from robot_human_interface.gui.research_widgets import (
    ExperimentPanel,
    PlaybackBar,
    ReadinessChecklist,
    SparklineWidget,
    SystemBanner,
    SystemBannerState,
)
from robot_human_interface.gui.runtime import (
    ReadinessReason,
    RobotReadiness,
    RobotUiState,
    RuntimeStatus,
)
from robot_human_interface.playback import PlaybackState


def _playback(*, eof: bool = False) -> PlaybackState:
    return PlaybackState(
        seekable=True,
        position_s=2.0,
        duration_s=10.0,
        frame_index=50,
        frame_count=250,
        fps=25.0,
        eof=eof,
    )


def test_playback_seek_is_committed_only_on_slider_release(qtbot) -> None:
    bar = PlaybackBar()
    qtbot.addWidget(bar)
    bar.set_playback_state(_playback(), session_state="PAUSED")
    spy = QSignalSpy(bar.seek_requested)

    bar.slider.setValue(50_000)
    assert len(spy) == 0
    bar.slider.sliderReleased.emit()

    assert len(spy) == 1
    assert spy[0][0] == pytest.approx(5.0)


def test_playback_live_disables_file_only_controls(qtbot) -> None:
    bar = PlaybackBar()
    qtbot.addWidget(bar)

    bar.set_playback_state(None, session_state="RUNNING", live=True)

    assert bar.mode_badge.text() == "LIVE"
    assert bar.play_button.isEnabled()
    assert not bar.slider.isEnabled()
    assert not bar.rate_combo.isEnabled()
    assert not bar.loop_checkbox.isEnabled()


def test_playback_emits_step_rate_and_valid_loop_range(qtbot) -> None:
    bar = PlaybackBar()
    qtbot.addWidget(bar)
    bar.set_playback_state(_playback(), session_state="PAUSED")
    step_spy = QSignalSpy(bar.step_requested)
    rate_spy = QSignalSpy(bar.rate_requested)
    loop_spy = QSignalSpy(bar.loop_requested)

    qtbot.mouseClick(bar.back_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(bar.forward_button, Qt.MouseButton.LeftButton)
    bar.rate_combo.setCurrentIndex(bar.rate_combo.findData(2.0))
    bar.slider.setValue(50_000)
    bar.slider.sliderReleased.emit()
    qtbot.mouseClick(bar.set_a_button, Qt.MouseButton.LeftButton)
    bar.slider.setValue(80_000)
    bar.slider.sliderReleased.emit()
    qtbot.mouseClick(bar.set_b_button, Qt.MouseButton.LeftButton)
    bar.loop_checkbox.setChecked(True)

    assert [entry[0] for entry in step_spy] == [-1, 1]
    assert rate_spy[-1][0] == 2.0
    assert loop_spy[-1] == [True, 5.0, 8.0]


def test_system_banner_armed_stop_is_immediate(qtbot) -> None:
    banner = SystemBanner()
    qtbot.addWidget(banner)
    spy = QSignalSpy(banner.stop_sending_requested)

    banner.show_armed(
        endpoint="ws://robot.local:8080",
        rate_hz=10.0,
        command_age_s=0.042,
        successful_sends=17,
    )
    qtbot.mouseClick(banner.stop_button, Qt.MouseButton.LeftButton)

    assert banner.banner_state is SystemBannerState.ARMED
    assert "42 мс" in banner.details_label.text()
    assert len(spy) == 1
    banner.clear()
    assert banner.isHidden()


def test_readiness_checklist_renders_authoritative_result(qtbot) -> None:
    checklist = ReadinessChecklist()
    qtbot.addWidget(checklist)
    readiness = RobotReadiness(
        ready=True,
        reason_code=ReadinessReason.READY,
        reason="",
        evaluated_at_s=5.0,
        runtime=RuntimeStatus.production(),
        pipeline_state="RUNNING",
        robot_state=RobotUiState.CONNECTED_DISARMED,
        source_id="reference:1",
        snapshot_sequence=10,
        snapshot_age_s=0.05,
        command_generation=3,
        safe_command_valid=True,
        free_base_active=True,
        balance_active=True,
    )

    checklist.set_readiness(readiness)

    assert checklist.readiness is readiness
    assert checklist.status_badge.text() == "ГОТОВО"
    assert checklist.rows["fresh"].value.text() == "50 мс"


def test_experiment_panel_emits_validated_spec_and_tracks_progress(qtbot) -> None:
    panel = ExperimentPanel()
    qtbot.addWidget(panel)
    panel.set_start_allowed(True)
    panel.participant_edit.setText("P001")
    panel.movement_edit.setText("Приседание")
    panel.method_combo.setCurrentText("baseline")
    panel.consent_checkbox.setChecked(True)
    panel.video_checkbox.setChecked(True)
    spy = QSignalSpy(panel.start_requested)

    qtbot.mouseClick(panel.start_button, Qt.MouseButton.LeftButton)

    assert len(spy) == 1
    spec = spy[0][0]
    assert isinstance(spec, ExperimentSpec)
    assert spec.participant_code == "P001"
    assert spec.record_video is True

    panel.set_recorder_state(
        RecorderState.RECORDING,
        run_id="run-1",
        accepted_samples=12,
        dropped_samples=1,
        elapsed_s=3.0,
    )
    assert panel.recording_active
    assert panel.samples_label.text() == "12"
    assert panel.drops_label.text() == "1"
    assert not panel.start_button.isEnabled()
    assert panel.stop_button.isEnabled()


def test_sparkline_keeps_only_ten_second_window(qtbot) -> None:
    sparkline = SparklineWidget()
    qtbot.addWidget(sparkline)

    sparkline.append(1.0, 0.0)
    sparkline.append(2.0, 5.0)
    sparkline.append(3.0, 11.0)

    assert sparkline.samples == ((5.0, 2.0), (11.0, 3.0))
