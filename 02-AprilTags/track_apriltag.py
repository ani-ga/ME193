"""Drive a LEGO Education Double Motor robot straight (no rotation, staying
parallel to the camera) to keep an AprilTag centered horizontally in the
webcam view."""

import cv2
import legoeducation as le

# Update these to match the Connection Card on your Double Motor.
CARD_COLOR = le.LEGO_COLOR_MAGENTA
CARD_SERIAL = '0998'

CAMERA_INDEX = 1
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

# Set to cv2.ROTATE_90_CLOCKWISE / COUNTERCLOCKWISE if the camera is ever
# mounted sideways; not needed for the current setup.
FRAME_ROTATION = None

DRIVE_SPEED_MAX = 60
DRIVE_SPEED_MIN = 25     # motors need a minimum PWM to actually overcome friction
CENTER_DEADBAND_PX = 20  # |error| below this counts as "centered" -> stop

# The two motors are mounted mirrored on the chassis, so the same commanded
# speed spins them in physically opposite directions. Copied from
# gesture_control.py's calibration for this robot; swap the -1 if the robot
# turns in place instead of driving straight.
MOTOR_DIRECTION_SIGN = {
    le.MOTOR_LEFT: 1,
    le.MOTOR_RIGHT: -1,
}
GLOBAL_DIRECTION_SIGN = -1

# Sign of the drive command that moves the robot toward a tag that's to its
# right in frame. Flip if the robot drives away from the tag instead of
# toward it.
DRIVE_DIRECTION_SIGN = -1


def run_motor(robot, motor, speed):
    robot.motor_run(motor=motor, speed=GLOBAL_DIRECTION_SIGN * speed * MOTOR_DIRECTION_SIGN[motor], blocking=False)


def drive(robot, speed):
    """Drive both wheels the same real-world direction (straight, no rotation).

    speed > 0 moves toward a tag on the right, < 0 toward one on the left.
    """
    speed = DRIVE_DIRECTION_SIGN * speed
    if speed == 0:
        robot.motor_stop(motor=le.MOTOR_BOTH, blocking=False)
        return
    # Same logical speed for both motors, same as gesture_control.py's
    # both_motors gesture: MOTOR_DIRECTION_SIGN already corrects for the
    # mirrored mounting, so this drives both wheels the same real-world
    # direction (straight) rather than spinning the chassis.
    run_motor(robot, le.MOTOR_LEFT, speed)
    run_motor(robot, le.MOTOR_RIGHT, speed)


def drive_speed_for_error(error_px, frame_width):
    """Proportional control: horizontal pixel error -> drive speed."""
    if abs(error_px) <= CENTER_DEADBAND_PX:
        return 0

    normalized = error_px / (frame_width / 2)  # -1..1
    speed = normalized * DRIVE_SPEED_MAX
    if abs(speed) < DRIVE_SPEED_MIN:
        speed = DRIVE_SPEED_MIN if speed > 0 else -DRIVE_SPEED_MIN
    return int(max(-DRIVE_SPEED_MAX, min(DRIVE_SPEED_MAX, speed)))


def main():
    robot = le.DoubleMotor()
    robot.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not robot.connected:
        print('Error connecting to Double Motor. Continuing in preview-only mode '
              '(the tag will be tracked but the robot will not move).')
        robot = None

    aruco_dict = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print('Error: could not open the camera. Check that no other app is using it and '
              'that this app has camera permission in System Settings > Privacy & Security > Camera.')
        if robot is not None:
            robot.disconnect()
        return

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                continue

            if FRAME_ROTATION is not None:
                frame = cv2.rotate(frame, FRAME_ROTATION)

            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width / 2

            corners, ids, _ = detector.detectMarkers(frame)

            speed = 0
            status = 'searching'
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                tag_corners = corners[0][0]
                tag_center_x = float(tag_corners[:, 0].mean())
                error_px = tag_center_x - frame_center_x
                speed = drive_speed_for_error(error_px, frame_width)
                status = 'centered' if speed == 0 else 'driving'

                cv2.circle(frame, (int(tag_center_x), int(frame_height / 2)), 6, (0, 255, 0), -1)

            cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), frame_height), (255, 0, 0), 1)

            if robot is not None:
                drive(robot, speed)

            cv2.putText(frame, f'status: {status} (speed={speed})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('AprilTag Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if robot is not None:
            robot.motor_stop(motor=le.MOTOR_BOTH)
            robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
