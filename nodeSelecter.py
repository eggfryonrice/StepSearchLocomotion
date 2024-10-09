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
    direction = np.cross(rightPos - spinePos, leftPos - spinePos)
    direction[1] = 0
    return normalize(direction)


class nodeSelecter:
    def __init__(self, folderPath, idleFilePath, contactVelocityThreshold=30):
        self.files: list[BVHFile] = []
        for fileName in os.listdir(folderPath):
            if fileName.endswith(".bvh"):
                filePath = os.path.join(folderPath, fileName)
                self.files.append(BVHFile(filePath))
        self.idleFile = BVHFile(idleFilePath)

        # file, start, end, startDirection, endDirection
        self.leftTransitions: list[tuple[BVHFile, int, int, np.ndarray, np.ndarray]] = (
            []
        )
        self.rightTransitions: list[
            tuple[BVHFile, int, int, np.ndarray, np.ndarray]
        ] = []

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
                    start = leftStartOfContacts[left]
                    end = rightStartOfContacts[right]
                    startJointsPosition = file.calculateJointsPositionFromFrame(start)
                    startDirection = getDirection(file, startJointsPosition)
                    endJointsPosition = file.calculateJointsPositionFromFrame(end)
                    endDirection = getDirection(file, endJointsPosition)
                    self.leftTransitions.append(
                        (file, start, end, startDirection, endDirection)
                    )
                    left += 1
                else:
                    start = rightStartOfContacts[right]
                    end = leftStartOfContacts[left]
                    startJointsPosition = file.calculateJointsPositionFromFrame(start)
                    startDirection = getDirection(file, startJointsPosition)
                    endJointsPosition = file.calculateJointsPositionFromFrame(end)
                    endDirection = getDirection(file, endJointsPosition)
                    self.rightTransitions.append(
                        (file, start, end, startDirection, endDirection)
                    )
                    right += 1

        self.leftContactIdx = self.idleFile.jointNames.index("LeftToe")
        self.rightContactIdx = self.idleFile.jointNames.index("RightToe")

        self.isLeftContact = True

    def getIdleNode(
        self,
        controlPosition,
        controlDirection,
    ):
        startJointsPosition = self.idleFile.calculateJointsPositionFromFrame(0)
        startDirection = getDirection(self.idleFile, startJointsPosition)
        rotation = quatToMat(vecToVecQuat(startDirection, controlDirection))
        position = toCartesian(
            self.idleFile.calculateJointPositionFromFrame(0, 0, rotation)
        )
        position[1] = 0
        translation = translationMat(controlPosition - position)
        return Node(
            self.idleFile, 0, self.idleFile.numFrames - 1, translation @ rotation
        )

    def getNextNode(
        self,
        currentJointsPosition,
        currentDirection,
        controlPosition,
        controlDirection,
    ):
        currentPosition = toCartesian(currentJointsPosition[0])
        currentPosition[1] = 0

        # if np.linalg.norm(currentPosition - controlPosition) < 25:
        #     return self.getIdleNode(controlPosition, controlDirection)

        transitions = (
            self.leftTransitions if self.isLeftContact else self.rightTransitions
        )
        contactIdx = self.leftContactIdx if self.isLeftContact else self.rightContactIdx

        objectiveDirection = normalize((controlPosition - currentPosition))

        bestIdx = 0
        bestError = float("inf")
        for idx in range(len(transitions)):
            file, start, end, startDirection, endDirection = transitions[idx]
            rotation = quatToMat(vecToVecQuat(startDirection, currentDirection))
            endDirection = toCartesian(rotation @ toProjective(endDirection))
            error = np.linalg.norm(endDirection - objectiveDirection)
            if error < bestError:
                bestError = error
                bestIdx = idx

        # bestIdx = 3 if self.isLeftContact else 23

        file, start, end, startDirection, _ = transitions[bestIdx]
        rotation = quatToMat(vecToVecQuat(startDirection, currentDirection))

        startContactPosition = toCartesian(
            file.calculateJointPositionFromFrame(contactIdx, start, rotation)
        )
        currentContactPosition = toCartesian(currentJointsPosition[contactIdx])
        translationVector = currentContactPosition - startContactPosition
        translationVector[1] = 0
        translation = translationMat(translationVector)

        self.isLeftContact = not self.isLeftContact
        return Node(file, start, end, translation @ rotation)


if __name__ == "__main__":
    print("nothing implemented")
