# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from unittest.mock import Mock

import pytest

scs = pytest.importorskip("scservo_sdk")

from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE  # noqa: E402


@pytest.mark.parametrize("failures", [1, 3])
def test_startup_lock_write_has_bounded_retries(failures):
    # Neither the real serial port nor a robot is opened. Only the packet handler is exercised.
    bus = FeetechMotorsBus("/dev/null", {"wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES)})
    bus.port_handler = Mock(is_open=True)
    bus.packet_handler = Mock()
    bus.packet_handler.getTxRxResult.return_value = "There is no status packet!"
    lock_address = STS_SMS_SERIES_CONTROL_TABLE["Lock"][0]
    lock_attempts = 0

    def write(port, motor_id, address, length, data):
        nonlocal lock_attempts
        if address == lock_address and data == [1]:
            lock_attempts += 1
            if lock_attempts <= failures:
                return scs.COMM_RX_TIMEOUT, 0
        return scs.COMM_SUCCESS, 0

    bus.packet_handler.writeTxRx.side_effect = write
    if failures == 1:
        with bus.torque_disabled(num_retry=2):
            pass
        assert lock_attempts == 2
    else:
        with pytest.raises(ConnectionError, match="after 3 tries"), bus.torque_disabled(num_retry=2):
            pass
        assert lock_attempts == 3
    bus.port_handler.openPort.assert_not_called()
    bus.port_handler.is_open = False


def test_negative_retry_count_rejected_before_writes():
    bus = FeetechMotorsBus("/dev/null", {"wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES)})
    bus.disable_torque = Mock()
    with pytest.raises(ValueError, match="nonnegative"), bus.torque_disabled(num_retry=-1):
        pass
    bus.disable_torque.assert_not_called()
