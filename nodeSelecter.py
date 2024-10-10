import numpy as np
import os
import random

from BVHFile import BVHFile
from transformationUtil import *


# node has file, startframe, endframe, and transformation
# to calculate joints positition from start to end of node
# additionally, it has requiredYRot and requiredTranslation that
# to be gradually applied during the motion
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


# find all contact points to get start of contacts
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


# get direction of movment by current joint posture
# get direction by cross product of two vectors:
# spine to left up leg, spine to right up leg
def getDirection(file: BVHFile, jointsPosition: np.ndarray):
    spinePos = toCartesian(jointsPosition[file.jointNames.index("Spine")])
    leftPos = toCartesian(jointsPosition[file.jointNames.index("LeftUpLeg")])
    rightPos = toCartesian(jointsPosition[file.jointNames.index("RightUpLeg")])
    direction = np.cross(rightPos - spinePos, leftPos - spinePos)
    direction[1] = 0
    return normalize(direction)


class nodeSelecter:
    def __init__(
        self,
        folderPath,
        idleFolderPath,
        rotationInterpolation: float = 0.0,
        translationInterpolation: float = 0.0,
        contactVelocityThreshold=30,
    ):
        self.files: list[BVHFile] = []
        for fileName in os.listdir(folderPath):
            if fileName.endswith(".bvh"):
                filePath = os.path.join(folderPath, fileName)
                self.files.append(BVHFile(filePath))
        self.idleFiles: list[BVHFile] = []
        for fileName in os.listdir(idleFolderPath):
            if fileName.endswith(".bvh"):
                filePath = os.path.join(idleFolderPath, fileName)
                self.idleFiles.append(BVHFile(filePath))
        self.file = self.files[0]

        # file, start, end, startDirection, endDirection
        self.leftTransitions: list[tuple[BVHFile, int, int, np.ndarray, np.ndarray]] = (
            []
        )
        self.rightTransitions: list[
            tuple[BVHFile, int, int, np.ndarray, np.ndarray]
        ] = []

        self.rotationInterpolation = rotationInterpolation
        self.translationInterpolation = translationInterpolation
        # find all nodes among files
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

        self.leftContactIdx = self.file.jointNames.index("LeftToe")
        self.rightContactIdx = self.file.jointNames.index("RightToe")

        self.isLeftContact = True

    # get start idle node by first position and direction
    def getStartIdleNode(
        self,
        controlPosition,
        controlDirection,
    ):
        idleIdx = random.randrange(len(self.idleFiles))
        file = self.idleFiles[idleIdx]
        startJointsPosition = file.calculateJointsPositionFromFrame(0)
        startDirection = getDirection(file, startJointsPosition)
        rotation = quatToMat(vecToVecQuat(startDirection, controlDirection))
        position = toCartesian(file.calculateJointPositionFromFrame(0, 0, rotation))
        position[1] = 0
        translation = translationMat(controlPosition - position)
        return Node(file, 0, file.numFrames - 1, translation @ rotation)

    # get idle node that can be smoothly interpolated from current motion
    def getIdleNode(
        self,
        jointsPosition,
    ):
        # select random idle motion from idle files
        idleIdx = random.randrange(len(self.idleFiles))
        file = self.idleFiles[idleIdx]

        direction = getDirection(file, jointsPosition)
        startJointsPosition = file.calculateJointsPositionFromFrame(0)
        startDirection = getDirection(file, startJointsPosition)
        rotation = quatToMat(vecToVecQuat(startDirection, direction))

        # determine which foot is at front
        rootPosition = toCartesian(jointsPosition[0])
        rootPosition[1] = 0
        leftContactPosition = toCartesian(jointsPosition[self.leftContactIdx])
        leftContactPosition[1] = 0
        rightContactPosition = toCartesian(jointsPosition[self.rightContactIdx])
        rightContactPosition[1] = 0
        leftDistance = np.linalg.norm(
            normalize(leftContactPosition - rootPosition) - direction
        )
        rightDistance = np.linalg.norm(
            normalize(rightContactPosition - rootPosition) - direction
        )

        frontContactIdx = (
            self.leftContactIdx
            if leftDistance < rightDistance
            else self.rightContactIdx
        )
        frontContactPosition = (
            leftContactPosition
            if leftDistance < rightDistance
            else rightContactPosition
        )

        # match front contact joint
        position = toCartesian(
            file.calculateJointPositionFromFrame(frontContactIdx, 0, rotation)
        )
        position[1] = 0
        translation = translationMat(frontContactPosition - position)
        return Node(file, 0, file.numFrames - 1, translation @ rotation)

    # get next node from current transition and left right contact state
    def getNextNode(
        self,
        currentJointsPosition,
        currentDirection,
        objectiveDirection,
    ):
        currentPosition = toCartesian(currentJointsPosition[0])
        currentPosition[1] = 0

        transitions = (
            self.leftTransitions if self.isLeftContact else self.rightTransitions
        )
        contactIdx = self.leftContactIdx if self.isLeftContact else self.rightContactIdx

        # find best index (best next node) and list of nodes that are good enough
        # good enough means rotation can be accurate when interpolated
        bestIdx = 0
        goodEnoughIdxes = []
        bestError = float("inf")
        for idx in range(len(transitions)):
            file, start, end, startDirection, endDirection = transitions[idx]
            rotation = quatToMat(vecToVecQuat(startDirection, currentDirection))
            endDirection = toCartesian(rotation @ toProjective(endDirection))
            error = np.linalg.norm(endDirection - objectiveDirection)
            if error < bestError:
                bestError = error
                bestIdx = idx
            if error < 2 * math.sin(self.rotationInterpolation * math.pi / 180 / 2):
                goodEnoughIdxes.append(idx)

        # if there is no good enough nodes we just ust bext index
        # else, we choose the one with biggest displacement along objective direction
        if len(goodEnoughIdxes) > 0:
            bestIdx = goodEnoughIdxes[0]
            bestDisplacement = -float("inf")
            for idx in goodEnoughIdxes:
                file, start, end, startDirection, endDirection = transitions[idx]
                rotation = quatToMat(vecToVecQuat(startDirection, currentDirection))
                startPosition = toCartesian(
                    file.calculateJointPositionFromFrame(0, start, rotation)
                )
                endPosition = toCartesian(
                    file.calculateJointPositionFromFrame(0, end, rotation)
                )
                displacement = np.dot(objectiveDirection, endPosition - startPosition)
                if displacement > bestDisplacement:
                    bestIdx = idx
                    bestDisplacement = displacement

        file, start, end, startDirection, endDirection = transitions[bestIdx]
        rotation = quatToMat(vecToVecQuat(startDirection, currentDirection))

        # rotation interpolation
        # we adjust end Direction to be matched with objective direction
        # maximum amount of adjustment is rotationInterpolation value in degree
        endDirection = toCartesian(rotation @ toProjective(endDirection))
        angleError = np.arccos(np.clip(np.dot(endDirection, objectiveDirection), -1, 1))
        interpAngle = np.clip(
            angleError,
            -self.rotationInterpolation * math.pi / 180,
            self.rotationInterpolation * math.pi / 180,
        )

        # translation interpolation
        # from startPosition and endPosition, we get displacement
        # we reduce orthogonal component of displacement with respect to objective direction
        # maximum by translationInterpolation Value
        startPosition = toCartesian(
            file.calculateJointPositionFromFrame(0, start, rotation)
        )
        startPosition[1] = 0
        endPosition = toCartesian(
            file.calculateJointPositionFromFrame(0, end, rotation)
        )
        endPosition[1] = 0
        translationError = orthogonalComponent(
            objectiveDirection, endPosition - startPosition
        )
        translationErrorSize = np.linalg.norm(translationError)
        interpTranslation = (
            -min(translationErrorSize, self.translationInterpolation)
            * translationError
            / translationErrorSize
        )

        # calculate translation by matching contact position
        startContactPosition = toCartesian(
            file.calculateJointPositionFromFrame(contactIdx, start, rotation)
        )
        currentContactPosition = toCartesian(currentJointsPosition[contactIdx])
        translationVector = currentContactPosition - startContactPosition
        translationVector[1] = 0
        translation = translationMat(translationVector)

        # revert contact state
        self.isLeftContact = not self.isLeftContact

        return Node(
            file, start, end, translation @ rotation, interpAngle, interpTranslation
        )


if __name__ == "__main__":
    print("nothing implemented")
