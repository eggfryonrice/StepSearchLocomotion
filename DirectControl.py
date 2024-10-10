import numpy as np

from transformationUtil import *
from pygameScene import pygameScene
from dataGenerater import dataGenerater
from nodeSelecter import nodeSelecter
from inertializationManager import inertializationManager


folderPaths = ["./walkingData", "./runningData"]
idleFolderPath = "./idleData"
dataFtn = dataGenerater(
    folderPaths,
    idleFolderPath,
    rotationInterpolation=5,
    translationInterpolation=7,
    contactVelocityThreshold=20,
)
file = dataFtn.file

scene = pygameScene(frameTime=file.frameTime, speed=0)
manager = inertializationManager(
    file,
    dataFtn.getNextData,
    halfLife=0.15,
    handleContact=True,
    unlockRadius=20,
    compare=False,
)

isMoving = False

while scene.running:
    eps = 0.1
    scene.cameraCenter[[0, 2]] = (
        scene.cameraCenter[[0, 2]] * (1 - eps)
        + toCartesian(dataFtn.currentJointsPosition[0])[[0, 2]] * eps
    )
    scene.highLightPoint = manager.currentRootPosition

    direction = scene.controlMovingDirection
    isMoving = scene.controlIsMoving
    mode = scene.mode % 2
    dataFtn.setObjective(direction, isMoving, mode)
    scene.updateScene(manager.getNextSceneInput())
