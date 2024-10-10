import numpy as np

from transformationUtil import *
from pygameScene import pygameScene
from nodeDataReader import nodeDataReader
from inertializationManager import inertializationManager


folderPaths = ["./walkingData", "./runningData"]
idleFilePath = "./idleData"
dataFtn = nodeDataReader(
    folderPaths, idleFilePath, interpolation=5, contactVelocityThreshold=20
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

    direction = scene.centerMovingDirection
    isMoving = scene.centerIsMoving
    mode = scene.mode % 2
    dataFtn.setObjective(direction, isMoving, mode)
    scene.updateScene(manager.getNextSceneInput())
