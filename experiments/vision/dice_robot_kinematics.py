import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def read_robot_parameters(ip):
    import websocket

    connection = websocket.create_connection(f"ws://{ip}:9090", timeout=10)
    try:
        values = {}
        for index, name in enumerate(("/robot_description", "/niryo_robot_vision/undistort_stream")):
            request_id = f"hand-eye-read-{index}"
            connection.send(json.dumps({
                "op": "call_service", "service": "/rosapi/get_param",
                "args": {"name": name}, "id": request_id,
            }))
            for _ in range(20):
                reply = json.loads(connection.recv())
                if reply.get("op") == "status" and reply.get("level") == "error":
                    raise ValueError(f"ROS read failed: {reply.get('msg')}")
                if reply.get("id") != request_id or reply.get("op") != "service_response":
                    continue
                if not reply.get("result"):
                    raise ValueError(f"Cannot read robot parameter {name}.")
                values[name] = json.loads(reply["values"]["value"])
                break
            else:
                raise ValueError(f"No matching ROS response for {name}.")
        return values
    finally:
        connection.close()


def vector(text, default):
    result = np.asarray([float(value) for value in (text or default).split()])
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Invalid URDF vector.")
    return result


def origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    result[:3, 3] = vector(element.get("xyz"), "0 0 0")
    roll, pitch, yaw = vector(element.get("rpy"), "0 0 0")
    result[:3, :3] = (
        cv2.Rodrigues(np.array([0, 0, yaw]))[0]
        @ cv2.Rodrigues(np.array([0, pitch, 0]))[0]
        @ cv2.Rodrigues(np.array([roll, 0, 0]))[0]
    )
    return result


class RobotKinematics:
    def __init__(self, description):
        root = ET.fromstring(description)
        self.description = description
        self.name = root.get("name")
        self.parents = {}
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            if child in self.parents:
                raise ValueError("URDF has duplicate child links.")
            self.parents[child] = joint
        joint6 = [joint for joint in self.parents.values() if joint.get("name") == "joint_6"]
        if len(joint6) != 1 or joint6[0].get("type") not in ("revolute", "continuous"):
            raise ValueError("Model must contain one revolute joint_6.")
        self.wrist_frame = joint6[0].find("parent").get("link")
        self.tool_frame = "tool_link"
        for frame in (self.wrist_frame, self.tool_frame):
            self.chain(frame)
        moving = [
            joint.get("name") for joint in self.chain(self.tool_frame)
            if joint.get("type") != "fixed"
        ]
        if moving != [f"joint_{index}" for index in range(1, 7)]:
            raise ValueError("Expected the six-joint serial chain ending at tool_link.")

    def chain(self, frame):
        result = []
        visited = set()
        while frame != "base_link":
            if frame in visited or frame not in self.parents:
                raise ValueError("No unambiguous URDF chain from base_link to " + frame)
            visited.add(frame)
            joint = self.parents[frame]
            result.append(joint)
            frame = joint.find("parent").get("link")
        return list(reversed(result))

    def forward(self, joints, frame):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Expected six finite robot joint angles.")
        angles = {f"joint_{index + 1}": angle for index, angle in enumerate(joints)}
        result = np.eye(4)
        for joint in self.chain(frame):
            result = result @ origin_transform(joint.find("origin"))
            kind = joint.get("type")
            if kind == "fixed":
                continue
            if kind not in ("revolute", "continuous") or joint.get("name") not in angles:
                raise ValueError("Unsupported movable joint in robot model.")
            axis_element = joint.find("axis")
            axis = vector(axis_element.get("xyz") if axis_element is not None else None, "1 0 0")
            length = np.linalg.norm(axis)
            if length < 1e-8:
                raise ValueError("URDF joint axis has zero length.")
            turn = np.eye(4)
            turn[:3, :3] = cv2.Rodrigues(axis / length * angles[joint.get("name")])[0]
            result = result @ turn
        return result

    def checked_wrist(self, joints, observed_tool):
        predicted = self.forward(joints, self.tool_frame)
        difference = np.linalg.inv(predicted) @ observed_tool
        angle = np.arccos(np.clip((np.trace(difference[:3, :3]) - 1) / 2, -1, 1))
        if np.linalg.norm(difference[:3, 3]) > 0.002 or angle > 0.02:
            raise ValueError(
                "Robot model does not match reported tool pose. Check TCP/frame settings; "
                "automatic calibration blocked."
            )
        return self.forward(joints, self.wrist_frame)
