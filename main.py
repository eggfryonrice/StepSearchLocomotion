from BVHFile import BVHFile
from pygameScene import pygameScene
from nodeDataReader import nodeDataReader
from inertializationManager import inertializationManager


folderPath = "./data"
idleFilePath = "./idle.bvh"
file = BVHFile(idleFilePath)

dataFtn = nodeDataReader(folderPath, idleFilePath)
scene = pygameScene(frameTime=file.frameTime)
manager = inertializationManager(
    file,
    dataFtn.getNextData,
    halfLife=0.15,
    handleContact=True,
    unlockRadius=30,
    compare=True,
)
while scene.running:
    position = scene.cameraCenter.copy()
    position[1] = 0
    direction = scene.centerMovingDirection
    isMoving = scene.centerIsMoving
    dataFtn.setObjective(position, direction, isMoving)
    scene.updateScene(manager.getNextSceneInput())
