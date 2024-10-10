import numpy as np

from transformationUtil import *
from pygameScene import pygameScene
from nodeDataReader import nodeDataReader
from inertializationManager import inertializationManager


folderPath = "./runningData"
idleFilePath = "./idleData"
dataFtn = nodeDataReader(folderPath, idleFilePath, interpolation=0)
file = dataFtn.file

scene = pygameScene(frameTime=file.frameTime, speed=0)
manager = inertializationManager(
    file,
    dataFtn.getNextData,
    halfLife=0.15,
    handleContact=True,
    unlockRadius=30,
    compare=False,
)

isMoving = False

while scene.running:
    eps = 0.1
    scene.cameraCenter[[0, 2]] = (
        scene.cameraCenter[[0, 2]] * (1 - eps)
        + toCartesian(dataFtn.currentJointsPosition[0])[[0, 2]] * eps
    )
    direction = scene.centerMovingDirection
    isMoving = scene.centerIsMoving
    dataFtn.setObjective(direction, isMoving)
    scene.updateScene(manager.getNextSceneInput())
