import numpy as np

from transformationUtil import *
from pygameScene import pygameScene
from nodeDataReader import nodeDataReader
from inertializationManager import inertializationManager


folderPath = "./walkingData"
idleFilePath = "./idleData"
dataFtn = nodeDataReader(folderPath, idleFilePath, interpolation=5)
file = dataFtn.file

scene = pygameScene(frameTime=file.frameTime, speed=200)
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
    cameraCenter = scene.cameraCenter.copy()
    cameraCenter[1] = 0
    position = toCartesian(dataFtn.currentJointsPosition[0])
    position[1] = 0
    direction = cameraCenter - position
    distance = np.linalg.norm(cameraCenter - position)
    if isMoving:
        isMoving = scene.centerIsMoving or (distance > 40)
    else:
        isMoving = scene.centerIsMoving
    dataFtn.setObjective(direction, isMoving)
    scene.updateScene(manager.getNextSceneInput())
