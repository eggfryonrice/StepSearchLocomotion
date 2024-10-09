import numpy as np

from BVHFile import BVHFile
from transformationUtil import *
from nodeSelecter import nodeSelecter, Node, getDirection


class nodeDataReader:
    def __init__(
        self,
        folderPath: str,
        idleFilePath: str,
        startPosition: np.ndarray = np.array([0, 0, 0]),
        startDirection: np.ndarray = np.array([0, 0, 1]),
        contactVelocityThreshold: int = 30,
    ):
        self.totalFrame: int = -1

        self.nodeSelecter = nodeSelecter(
            folderPath, idleFilePath, contactVelocityThreshold
        )
        self.file = self.nodeSelecter.files[0]
        self.node: Node = self.nodeSelecter.getStartIdleNode(
            startPosition, startDirection
        )
        self.currFrame: int = self.node.startFrame

        self.contactVelocityThreshold = contactVelocityThreshold

        self.currentJointsPosition = None
        self.currentDirection = startDirection
        self.objectivePosition = startPosition
        self.objectiveDirection = startDirection
        self.objectiveIsMoving = False

        self.idle = True
        self.idleJointsPosition = self.node.file.calculateJointsPositionFromFrame(
            self.node.startFrame, self.node.transformation
        )

    def setObjective(
        self,
        objectivePosition: np.ndarray,
        objectiveDirection: np.ndarray,
        isMoving: bool,
    ):
        self.objectivePosition = objectivePosition
        self.objectiveDirection = objectiveDirection
        self.objectiveIsMoving = isMoving

    def getNextData(self):
        discontinuity: bool = self.currFrame == self.node.endFrame
        self.totalFrame += 1 - discontinuity

        if self.currFrame > self.node.endFrame:
            if self.idle:
                self.node = self.nodeSelecter.getIdleNode(self.idleJointsPosition)
            else:
                self.node = self.nodeSelecter.getNextNode(
                    self.currentJointsPosition,
                    self.currentDirection,
                    self.objectivePosition,
                    self.objectiveDirection,
                )
            self.currFrame = self.node.startFrame

        contactIdx1 = self.file.jointNames.index("LeftToe")
        c1 = (
            self.node.file.getJointSpeed(contactIdx1, self.currFrame)
            < self.contactVelocityThreshold
        )
        contactIdx2 = self.file.jointNames.index("RightToe")
        c2 = (
            self.node.file.getJointSpeed(contactIdx2, self.currFrame)
            < self.contactVelocityThreshold
        )

        translationData = self.node.file.translationDatas[self.currFrame]
        quatData = eulersToQuats(self.node.file.eulerDatas[self.currFrame])

        # adjust data to transformation
        translationData = toCartesian(
            self.node.transformation
            @ toProjective(
                translationData + toCartesian(self.node.file.jointOffsets[0])
            )
        ) - toCartesian(self.node.file.jointOffsets[0])
        quatData[0] = multQuat(matToQuat(self.node.transformation), quatData[0])

        # adjust data to required extra transformation
        translationData += (
            self.node.requiredTranslation
            * (self.currFrame - self.node.startFrame)
            / (self.node.endFrame - self.node.startFrame)
        )
        yRotation = (
            self.node.requiredYRot
            * (self.currFrame - self.node.startFrame)
            / (self.node.endFrame - self.node.startFrame)
        )
        quatData[0] = multQuat(quatY(yRotation), quatData[0])

        jointsPosition = self.file.calculateJointsPositionFromQuaternionData(
            translationData, quatData
        )
        self.currentDirection = getDirection(self.file, jointsPosition)
        self.currentJointsPosition = jointsPosition

        rootPosition = toCartesian(self.currentJointsPosition[0])
        rootPosition[1] = 0
        distance = np.linalg.norm(rootPosition - self.objectivePosition)
        if (not self.objectiveIsMoving) and (not self.idle) and (distance < 30):
            self.idle = True
            self.idleJointsPosition = self.currentJointsPosition
            discontinuity = True
            self.currFrame = self.node.endFrame
        if self.idle and self.objectiveIsMoving:
            self.idle = False
            discontinuity = True
            self.currFrame = self.node.endFrame

        self.currFrame += 1
        return (
            self.totalFrame,
            translationData,
            quatData,
            np.array([c1, c2]),
            discontinuity,
        )
