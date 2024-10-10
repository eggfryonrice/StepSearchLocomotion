import pygame
from pygame.locals import *  # type: ignore
from OpenGL.GL import *  # type: ignore
from OpenGL.GLU import *  # type: ignore
from OpenGL.GLUT import *  # type: ignore
import numpy as np
import math

# list of (ndarray of 3d points and color)
jointsPositionsInput = list[tuple[np.ndarray, tuple[float, float, float]]]
# list of (list of (fromPoint, toPoint, link rotation quaternion) and color)
linkssInput = list[
    tuple[
        list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        tuple[float, float, float],
    ]
]

sceneInput = tuple[
    int,
    jointsPositionsInput,
    linkssInput,
]


class pygameScene:
    def __init__(
        self,
        frameTime: float = 0.033,
        cameraAngleX: float = math.pi / 8,
        cameraAngleY: float = -math.pi / 2,
        speed: float = 150,
        width: int = 2560,
        height: int = 1600,
        sphereRadius: float = 3,
        cuboidWidth: float = 4,
    ):
        pygame.init()
        self.running = True

        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        self.width = width
        self.height = height

        self.clock = pygame.time.Clock()
        self.frameTime = frameTime

        glutInit()
        self.initOpengl()
        self.initLighting()
        self.lightPosition = [0, 1000, 0, 0]

        self.sphereRadius: float = sphereRadius
        self.cuboidWidth: float = cuboidWidth

        self.chessBoardCenter: np.ndarray = np.array([0, 0, 0])

        # Camera parameters
        self.cameraCenter: np.ndarray = np.array([0.0, 0.0, 0.0])
        # cameracenter will be initialized by first data
        self.cameraCenterInitializedByFirstData = False
        self.cameraAngleX = cameraAngleX
        self.cameraAngleY = cameraAngleY
        self.cameraDistance = 500
        self.speed = speed

        self.mouseDragging: bool = False
        self.prevMousePosition: tuple[int, int] = (0, 0)

        self.centerMovingDirection = np.array([0, 0, 1])
        self.centerIsMoving = False

        self.mode = 0

        self.highLightPoint = np.array([0.0, 0.0, 0.0])

    def initOpengl(self):
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def initLighting(self):
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        # Define light properties
        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        material_specular = [0.1, 0.1, 0.1, 1.0]
        material_shininess = [10.0]
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material_specular)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material_shininess)

    def initCameraCenter(self, jointsPositions: jointsPositionsInput):
        if len(jointsPositions) == 0:
            return
        self.cameraCenter = jointsPositions[0][0][0]
        for jointsPosition, _ in jointsPositions:
            for jointPosition in jointsPosition:
                if self.cameraCenter[1] > jointPosition[1]:
                    self.cameraCenter[1] = jointPosition[1]
        self.chessBoardCenter = self.cameraCenter.copy()
        self.cameraCenterInitializedByFirstData = True

    def handleMouseInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if not self.mouseDragging:
                        self.prevMousePosition = pygame.mouse.get_pos()
                    self.mouseDragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouseDragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouseDragging:
                    mouseX, mouseY = pygame.mouse.get_pos()
                    xrot = (mouseY - self.prevMousePosition[1]) * math.pi / self.height
                    yrot = (mouseX - self.prevMousePosition[0]) * math.pi / self.width
                    self.cameraAngleX = max(
                        0, min(math.pi / 2 - 1e-8, self.cameraAngleX + xrot)
                    )
                    self.cameraAngleY = self.cameraAngleY + yrot

                    self.prevMousePosition = (mouseX, mouseY)

            elif event.type == pygame.MOUSEWHEEL:
                self.cameraDistance += event.y * 30
                if self.cameraDistance < 20:
                    self.cameraDistance = 20

    def handleKeyBoardInput(self):
        keys = pygame.key.get_pressed()

        direction = np.array([0.0, 0.0, 0.0])
        if keys[pygame.K_UP]:
            direction[0] -= math.cos(self.cameraAngleY)
            direction[2] -= math.sin(self.cameraAngleY)
        if keys[pygame.K_DOWN]:
            direction[0] += math.cos(self.cameraAngleY)
            direction[2] += math.sin(self.cameraAngleY)
        if keys[pygame.K_LEFT]:
            direction[0] -= math.sin(self.cameraAngleY)
            direction[2] += math.cos(self.cameraAngleY)
        if keys[pygame.K_RIGHT]:
            direction[0] += math.sin(self.cameraAngleY)
            direction[2] -= math.cos(self.cameraAngleY)
        if np.linalg.norm(direction) > 1e-8:
            direction = direction / np.linalg.norm(direction)
            self.centerMovingDirection = direction
            self.centerIsMoving = True
            change = self.speed * self.frameTime * direction
            self.cameraCenter += change
        else:
            self.centerIsMoving = False

        # Handle arrow keys for camera panning (left/right/up/down)
        if keys[pygame.K_a]:  # Rotate camera left
            self.cameraAngleY -= 0.05
        if keys[pygame.K_d]:  # Rotate camera right
            self.cameraAngleY += 0.05
        if keys[pygame.K_w]:  # Rotate camera left
            self.cameraAngleX = min(math.pi / 2 - 1e-8, self.cameraAngleX + 0.05)
        if keys[pygame.K_s]:  # Rotate camera right
            self.cameraAngleX = max(0, self.cameraAngleX - 0.05)

        if keys[pygame.K_0]:
            self.mode = 0
        if keys[pygame.K_1]:
            self.mode = 1
        if keys[pygame.K_2]:
            self.mode = 2
        if keys[pygame.K_3]:
            self.mode = 3

        glLoadIdentity()

    def adjustCamera(self):
        glLoadIdentity()

        # rotate camera along x axis first, and then along y axis
        cameraY = self.cameraDistance * math.sin(self.cameraAngleX)
        cameraX = (
            self.cameraDistance
            * math.cos(self.cameraAngleX)
            * math.cos(self.cameraAngleY)
        )
        cameraZ = (
            self.cameraDistance
            * math.cos(self.cameraAngleX)
            * math.sin(self.cameraAngleY)
        )
        cx, cy, cz = self.cameraCenter
        gluLookAt(cx + cameraX, cy + cameraY, cz + cameraZ, cx, cy, cz, 0, 1, 0)

        glLightfv(GL_LIGHT0, GL_POSITION, self.lightPosition)

    def drawSphere(self, position, color=(0.5, 0.5, 1.0)):
        glPushMatrix()
        glTranslatef(position[0], position[1], position[2])
        glColor3f(color[0], color[1], color[2])
        glutSolidSphere(self.sphereRadius, 20, 20)
        glPopMatrix()

    def drawCuboid(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        rotationQuat: np.ndarray,
        color: tuple[float, float, float] = (1.0, 0.5, 0.5),
    ):
        glPushMatrix()

        mid_point = (start_pos + end_pos) / 2
        direction = end_pos - start_pos
        length = max(float(np.linalg.norm(direction) - 2 * self.sphereRadius), 0.0)

        glTranslatef(mid_point[0], mid_point[1], mid_point[2])

        angle = 2 * np.arccos(rotationQuat[0])
        axis = rotationQuat[1:]
        if np.linalg.norm(axis) > 1e-3:
            glRotatef(angle * 180 / math.pi, axis[0], axis[1], axis[2])

        glScalef(self.cuboidWidth, length, self.cuboidWidth)

        glColor3f(color[0], color[1], color[2])

        # Draw the cuboid with normals
        vertices = self.cube_faces()
        normals = self.cube_normals()

        glBegin(GL_QUADS)
        for i, face in enumerate(vertices):
            glNormal3fv(normals[i])  # Set the normal for each face
            for vertex in face:
                glVertex3fv(vertex)
        glEnd()

        glPopMatrix()

    def cube_faces(self):
        vertices = [
            [0.5, -0.5, -0.5],  # Front bottom right
            [0.5, 0.5, -0.5],  # Front top right
            [-0.5, 0.5, -0.5],  # Front top left
            [-0.5, -0.5, -0.5],  # Front bottom left
            [0.5, -0.5, 0.5],  # Back bottom right
            [0.5, 0.5, 0.5],  # Back top right
            [-0.5, -0.5, 0.5],  # Back bottom left
            [-0.5, 0.5, 0.5],  # Back top left
        ]

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front face
            [vertices[5], vertices[4], vertices[6], vertices[7]],  # Back face
            [vertices[3], vertices[2], vertices[7], vertices[6]],  # Left face
            [vertices[1], vertices[0], vertices[4], vertices[5]],  # Right face
            [vertices[2], vertices[1], vertices[5], vertices[7]],  # Top face
            [vertices[0], vertices[3], vertices[6], vertices[4]],  # Bottom face
        ]
        return faces

    def cube_normals(self):
        normals = [
            [0, 0, -1],  # Front face normal
            [0, 0, 1],  # Back face normal
            [-1, 0, 0],  # Left face normal
            [1, 0, 0],  # Right face normal
            [0, 1, 0],  # Top face normal
            [0, -1, 0],  # Bottom face normal
        ]
        return normals

    def drawArrow(
        self, startPoint, direction, length=100, lineWidth=20, arrowHeadSize=50
    ):
        height = self.chessBoardCenter[1] - self.sphereRadius
        endPoint = startPoint + direction * length

        lineEndPoint = startPoint + direction * (length - arrowHeadSize)

        glLineWidth(lineWidth)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(startPoint[0], height + 1, startPoint[2])
        glVertex3f(lineEndPoint[0], height + 1, lineEndPoint[2])
        glEnd()

        perpendicular_dir = np.cross(direction, np.array([0, 1, 0]))
        perpendicular_dir = perpendicular_dir / np.linalg.norm(perpendicular_dir)

        arrow_head_point1 = (
            endPoint - direction * arrowHeadSize + perpendicular_dir * arrowHeadSize / 2
        )
        arrow_head_point2 = (
            endPoint - direction * arrowHeadSize - perpendicular_dir * arrowHeadSize / 2
        )

        glBegin(GL_TRIANGLES)
        glVertex3f(endPoint[0], height + 1, endPoint[2])  # Arrowhead tip
        glVertex3f(arrow_head_point1[0], height + 1, arrow_head_point1[2])
        glVertex3f(arrow_head_point2[0], height + 1, arrow_head_point2[2])
        glEnd()

    def drawChessBoard(
        self, numGrid: int = 14, blockSize: float = 50, drawHighLight: bool = False
    ):
        # floor is located at cameracenterheight - joint radius
        height = self.chessBoardCenter[1] - self.sphereRadius

        glPushMatrix()

        halfSize = (numGrid * blockSize) / 2

        for i in range(numGrid):
            for j in range(numGrid):
                x = -halfSize + i * blockSize + self.chessBoardCenter[0]
                z = -halfSize + j * blockSize + self.chessBoardCenter[2]

                if (i + j) % 2 == 0:
                    glColor3f(0.9, 0.9, 0.9)
                else:
                    glColor3f(0.1, 0.1, 0.1)

                glBegin(GL_QUADS)
                glVertex3f(x, height, z)
                glVertex3f(x + blockSize, height, z)
                glVertex3f(x + blockSize, height, z + blockSize)
                glVertex3f(x, height, z + blockSize)
                glEnd()

        if drawHighLight:
            x = self.highLightPoint[0]
            z = self.highLightPoint[2]
            radius = 10
            segments = 30
            glColor3f(1.0, 0.0, 0.0)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(x, height + 1, z)
            for i in range(segments + 1):
                angle = 2 * math.pi * i / segments
                dx = radius * math.cos(angle)
                dz = radius * math.sin(angle)
                glVertex3f(x + dx, height + 1, z + dz)
            glEnd()
            self.drawArrow(self.highLightPoint, self.centerMovingDirection)

        glPopMatrix()

    # draw elapsed time on top right
    def drawElapsedTimeAndFrame(self, frame: int) -> None:
        font = pygame.font.Font(None, 50)
        elapsedSurface = font.render(
            f"Time: {frame * self.frameTime:.2f}s, Frame: {frame}",
            True,
            (255, 255, 255),
            (0, 0, 0),
        )

        text_data = pygame.image.tostring(elapsedSurface, "RGBA", True)
        width, height = elapsedSurface.get_size()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glRasterPos2d(self.width - width - 10, self.height - 50)
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def updateScene(self, objects: sceneInput):
        frame, jointsPositions, linkss = objects

        if not self.cameraCenterInitializedByFirstData:
            self.initCameraCenter(jointsPositions)

        self.handleKeyBoardInput()
        self.handleMouseInput()
        self.adjustCamera()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore

        if not self.running:
            return

        self.drawChessBoard(drawHighLight=True)

        for jointsPosition, color in jointsPositions:
            for jointPosition in jointsPosition:
                self.drawSphere(jointPosition, color=color)

        for links, color in linkss:
            for link in links:
                self.drawCuboid(link[0], link[1], link[2], color=color)

        self.drawElapsedTimeAndFrame(frame)

        pygame.display.flip()

        self.clock.tick(1 / self.frameTime)


if __name__ == "__main__":
    scene = pygameScene()
    while scene.running:
        scene.updateScene((0, [(np.array([[0.0, 0.0, 0.0]]), (1, 0.5, 0.5))], []))
