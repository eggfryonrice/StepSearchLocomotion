import numpy as np
from typing import Callable

from BVHFile import BVHFile
from pygameScene import pygameScene, sceneInput
from contactManager import contactManager
from transformationUtil import *

# frame, translationData, eulerData, contactState, discontinuity flag
inertializationManagerInput = tuple[int, np.ndarray, np.ndarray, np.ndarray, bool]


class inertializationManager:
    def __init__(
        self,
        file: BVHFile,
        dataFtn: Callable[[], inertializationManagerInput],
        halfLife: float = 0.15,
        compare: bool = False,
        handleContact: bool = True,
        contactJointNames=["LeftToe", "RightToe"],
        contactHalfLife: float = 0.15,
        unlockRadius: float = 20,
        footHeight: float = 2,
    ):
        self.file: BVHFile = file

        self.dataFtn = dataFtn

        self.halfLife = halfLife

        self.currentData: inertializationManagerInput = self.dataFtn()
        self.nextData: inertializationManagerInput = self.dataFtn()
        self.previousData: inertializationManagerInput = self.currentData

        self.translationOffset = np.array([0, 0, 0])
        self.translationVelocityOffset = np.array([0, 0, 0])

        self.jointsQuatOffset = np.array(
            [np.array([1, 0, 0, 0]) for _ in range(self.file.numJoints)]
        )
        self.jointsQuatVelocityOffset = np.array(
            [np.array([0, 0, 0]) for _ in range(self.file.numJoints)]
        )

        self.compare = compare

        self.handleContact = handleContact
        self.contactManager = contactManager(
            self.file,
            contactJointNames=contactJointNames,
            unlockRadius=unlockRadius,
            footHeight=footHeight,
            halfLife=contactHalfLife,
        )

        self.currentRootPosition = toCartesian(
            self.file.calculateJointPositionFromQuaternionData(
                0, self.currentData[1], self.currentData[2]
            )
        )

    def dampOffsets(self):
        y = 2 * 0.6931 / self.halfLife
        eydt = np.exp(-y * self.file.frameTime)

        j1 = self.translationVelocityOffset + self.translationOffset * y

        self.translationOffset = eydt * (
            self.translationOffset + j1 * self.file.frameTime
        )
        self.translationVelocityOffset = eydt * (
            self.translationVelocityOffset - j1 * y * self.file.frameTime
        )

        j0 = quatsToScaledAngleAxises(self.jointsQuatOffset)
        j1 = self.jointsQuatVelocityOffset + j0 * y
        self.jointsQuatOffset = scaledAngleAxisesToQuats(
            eydt * (j0 + j1 * self.file.frameTime)
        )
        self.jointsQuatVelocityOffset = eydt * (
            self.jointsQuatVelocityOffset - j1 * y * self.file.frameTime
        )

    # discontinuity signs is true on last frame before discontinuity
    # data is given as following. when discontinuity happens at frame 3, data order is
    # 1, 2, 3, 3, 4, 5, ....
    # where first 3 is motion before discontinuity at frame 3,
    # and second 3 is motion after discontinuity at frame 3
    def manageInertialization(self) -> tuple[int, np.ndarray, np.ndarray]:
        frame, currTranslationData, currQuatData, _, discontinuity = self.currentData

        self.dampOffsets()
        translationData = currTranslationData + self.translationOffset
        quatData = multQuats(self.jointsQuatOffset, currQuatData)

        # in normal case, return offset considered joint position
        if not discontinuity:
            self.previousData = self.currentData
            self.currentData = self.nextData
            self.nextData = self.dataFtn()
            return frame, translationData, quatData

        # on discontinuity, we need information for two frames prior to discontinuity,
        # two frames after discontinuity
        _, prevTranslationData, prevQuatData, _, _ = self.previousData
        frame, nextTranslationData, nextQuatData, _, _ = self.nextData
        nnextData = self.dataFtn()
        _, nnextTranslationData, nnextQuatData, _, _ = nnextData

        # calculate current source root Position and velocity,
        # joints quat and quatVelocity
        prevRootPosition = toCartesian(
            self.file.calculateJointPositionFromQuaternionData(
                0, prevTranslationData, prevQuatData
            )
        )
        currRootPosition = toCartesian(
            self.file.calculateJointPositionFromQuaternionData(
                0, currTranslationData, currQuatData
            )
        )
        currRootVelocity = (currRootPosition - prevRootPosition) / self.file.frameTime

        currQuatVelocity = (
            quatsToScaledAngleAxises(multQuats(currQuatData, invQuats(prevQuatData)))
            / self.file.frameTime
        )

        # calculate next source root Position and velocity,
        # joints quat and quatVelocity
        nextRootPosition = toCartesian(
            self.file.calculateJointPositionFromEulerData(
                0, nextTranslationData, nextQuatData
            )
        )
        nnextRootPosition = toCartesian(
            self.file.calculateJointPositionFromEulerData(
                0, nnextTranslationData, nnextQuatData
            )
        )

        nextRootVelocity = (nnextRootPosition - nextRootPosition) / self.file.frameTime

        nextQuatVelocity = (
            quatsToScaledAngleAxises(multQuats(nnextQuatData, invQuats(nextQuatData)))
            / self.file.frameTime
        )

        # update offset
        self.translationOffset = (
            self.translationOffset + currRootPosition - nextRootPosition
        )
        self.translationVelocityOffset = (
            self.translationVelocityOffset + currRootVelocity - nextRootVelocity
        )
        self.jointsQuatOffset = absQuats(
            multQuats(
                multQuats(self.jointsQuatOffset, currQuatData),
                invQuats(nextQuatData),
            )
        )

        self.jointsQuatVelocityOffset = (
            self.jointsQuatVelocityOffset + currQuatVelocity - nextQuatVelocity
        )

        self.previousData = self.nextData
        self.currentData = nnextData
        self.nextData = self.dataFtn()
        return frame, translationData, quatData

    def getNextSceneInput(self) -> sceneInput:
        jointsPositions = []
        linkss = []
        _, currTranslationData, currQuatData, contact, _ = self.currentData
        frame, translationData, quatData = self.manageInertialization()

        if self.handleContact:
            translationData, quatData = self.contactManager.manageContact(
                (translationData, quatData, contact)
            )

        adjustedJointsPosition, adjustedLinks = (
            self.file.calculateJointsPositionAndLinksFromQuaternionData(
                translationData, quatData
            )
        )
        jointsPositions.append((adjustedJointsPosition, (1.0, 0.5, 0.5)))
        linkss.append((adjustedLinks, (0.5, 0.5, 1.0)))

        if self.compare:
            originalTranslation = currTranslationData + self.contactManager.translation
            originalQuat = currQuatData
            originalJointsPosition, originalLinks = (
                self.file.calculateJointsPositionAndLinksFromQuaternionData(
                    originalTranslation, originalQuat
                )
            )
            jointsPositions.append((originalJointsPosition, (1.0, 0.0, 0.0)))
            linkss.append((originalLinks, (1.0, 0.0, 0.0)))

        self.currentRootPosition = adjustedJointsPosition[0]

        return (
            frame,
            jointsPositions,
            linkss,
        )
