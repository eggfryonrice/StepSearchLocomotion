import numpy as np
import os

from BVHFile import BVHFile
from transformationUtil import *


class Node:
    def __init__(
        self,
        file: BVHFile,
        startFrame: int,
        endFrame: int,
        transformation: np.ndarray = np.eye(4),
        requiredYRot: float = 0,
        requiredTranslation: np.ndarray = np.array([0, 0, 0]),
    ):
        self.file: BVHFile = file
        self.startFrame: int = startFrame
        self.endFrame: int = endFrame
        self.transformation = transformation
        self.requiredYRot = requiredYRot
        self.requiredTranslation = requiredTranslation


def findStartOfContacts(
    file: BVHFile, contactJointName="LeftToe", velocityThreshold=30
):
    contacts = []
    contactJointIdx = file.jointNames.index(contactJointName)

    for i in range(file.numFrames):
        speed = file.getJointSpeed(contactJointIdx, i)
        contacts.append(speed < velocityThreshold)

    startOfContacts = [
        i for i in range(1, len(contacts)) if contacts[i] and not contacts[i - 1]
    ]

    return startOfContacts


def getDirection(file: BVHFile, jointsPosition: np.ndarray):
    spinePos = toCartesian(jointsPosition[file.jointNames.index("Spine")])
    leftPos = toCartesian(jointsPosition[file.jointNames.index("LeftUpLeg")])
    rightPos = toCartesian(jointsPosition[file.jointNames.index("RightUpLeg")])
    direction = normalize(np.cross(rightPos - spinePos, leftPos - spinePos)[[0, 2]])
    return direction


class nodeSelecter:
    def __init__(self, folderPath, idleFilePath, contactVelocityThreshold=30):
        self.files: list[BVHFile] = []
        for fileName in os.listdir(folderPath):
            if fileName.endswith(".bvh"):
                filePath = os.path.join(folderPath, fileName)
                self.files.append(BVHFile(filePath))

        self.leftTransition: list[tuple[BVHFile, int, int]] = []
        self.rightTransition: list[tuple[BVHFile, int, int]] = []

        for file in self.files:
            leftStartOfContacts = findStartOfContacts(
                file, "LeftToe", contactVelocityThreshold
            )
            rightStartOfContacts = findStartOfContacts(
                file, "RightToe", contactVelocityThreshold
            )
            left, right = 0, 0
            while left < len(leftStartOfContacts) and right < len(rightStartOfContacts):
                if leftStartOfContacts[left] < rightStartOfContacts[right]:
                    self.leftTransition.append(
                        (file, leftStartOfContacts[left], rightStartOfContacts[right])
                    )
                    left += 1
                else:
                    self.rightTransition.append(
                        (file, rightStartOfContacts[right], leftStartOfContacts[left])
                    )
                    right += 1

        self.idleFile = BVHFile(idleFilePath)

        self.isLeftContact = True

    def getFirstIdleNode(self, startPosition: np.ndarray, startDirection: np.ndarray):
        jointsPosition = self.idleFile.calculateJointsPositionFromFrame(0)
        direction = getDirection(self.idleFile, jointsPosition)
        rotation = quatToMat(
            vecToVecQuat(
                np.array([direction[0], 0, direction[1]]),
                np.array([startDirection[0], 0, startDirection[1]]),
            )
        )

        position = toCartesian(
            self.idleFile.calculateJointPositionFromFrame(0, 0, rotation)
        )[[0, 2]]
        translation = translationMat(
            np.array(
                [startPosition[0] - position[0], 0, startPosition[1] - position[1]]
            )
        )
        return Node(
            self.idleFile, 0, self.idleFile.numFrames - 1, translation @ rotation
        )

    def getNextNode(
        self,
        currentJointsPosition,
        currentDirection,
        objectivePosition,
        objectiveDirection,
    ):
        currentPosition = currentJointsPosition[0][[0, 2]]
        if (
            np.linalg.norm(currentPosition - objectivePosition) < 20
            and np.linalg.norm(currentDirection - objectiveDirection) < 0.5
        ):
            jointsPosition = self.idleFile.calculateJointsPositionFromFrame(0)
            direction = getDirection(self.idleFile, jointsPosition)
            rotation = quatToMat(
                vecToVecQuat(
                    np.array([direction[0], 0, direction[1]]),
                    np.array([objectiveDirection[0], 0, objectiveDirection[1]]),
                )
            )

            position = toCartesian(
                self.idleFile.calculateJointPositionFromFrame(0, 0, rotation)
            )[[0, 2]]
            translation = translationMat(
                np.array(
                    [
                        objectivePosition[0] - position[0],
                        0,
                        objectivePosition[1] - position[1],
                    ]
                )
            )
            return Node(
                self.idleFile, 0, self.idleFile.numFrames - 1, translation @ rotation
            )

        if self.isLeftContact:
            idx = 13
            file, start, end = self.leftTransition[idx]
            jointsPosition = file.calculateJointsPositionFromFrame(start)
            direction = getDirection(file, jointsPosition)
            rotation = quatToMat(
                vecToVecQuat(
                    np.array([direction[0], 0, direction[1]]),
                    np.array([currentDirection[0], 0, currentDirection[1]]),
                )
            )
            leftToePosition = toCartesian(
                file.calculateJointPositionFromFrame(
                    file.jointNames.index("LeftToe"), start, rotation
                )
            )
            currentLeftToePosition = toCartesian(
                currentJointsPosition[file.jointNames.index("LeftToe")]
            )
            translationVector = currentLeftToePosition - leftToePosition
            translationVector[1] = 0
            translation = translationMat(translationVector)
            self.isLeftContact = False
            return Node(file, start, end, translation @ rotation)
        else:
            idx = 25
            file, start, end = self.rightTransition[idx]
            jointsPosition = file.calculateJointsPositionFromFrame(start)
            direction = getDirection(file, jointsPosition)
            rotation = quatToMat(
                vecToVecQuat(
                    np.array([direction[0], 0, direction[1]]),
                    np.array([currentDirection[0], 0, currentDirection[1]]),
                )
            )

            rightToePosition = toCartesian(
                file.calculateJointPositionFromFrame(
                    file.jointNames.index("RightToe"), start, rotation
                )
            )
            currentRightToePosition = toCartesian(
                currentJointsPosition[file.jointNames.index("RightToe")]
            )
            translationVector = currentRightToePosition - rightToePosition
            translationVector[1] = 0
            translation = translationMat(translationVector)
            self.isLeftContact = True
            return Node(file, start, end, translation @ rotation)


if __name__ == "__main__":
    nodeSelecter("./data")
