from PIL import Image
import math
import random

# Constants
Cw, Ch = 500, 500       # Canvas width and height (pixels)
Vw, Vh = 2.0, 2.0         # Viewport width and height (scene units)
d = 1.0                  # Distance from camera to projection plane

# Colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
PINK = (255, 105, 180)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
BROWN = (139, 69, 19)

COLORS = [BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW, ORANGE, PURPLE, PINK, WHITE, GRAY, BROWN]

# Create a new image (1000x500 pixels, RGB mode, black background)
img = Image.new('RGB', (500, 500), color='black')
pixels = img.load()

def Interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i1 - i0):
        values.append(d)
        d = d + a
    return values


def DrawLine(P0, P1, color):
    x0, y0 = P0
    x1, y1 = P1

    if abs(x1 - x0) > abs(y1 - y0):
        # Line is horizontal-ish
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0

        ys = Interpolate(x0, y0, x1, y1)
        for x in range(x0, x1):
            y = int(ys[x - x0])
            if 0 <= x < img.width and 0 <= y < img.height:
                pixels[x, y] = color
    else:
        # Line is vertical-ish
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0

        xs = Interpolate(y0, x0, y1, x1)
        for y in range(y0, y1):
            x = int(xs[y - y0])
            if 0 <= x < img.width and 0 <= y < img.height:
                pixels[x, y] = color

def ProjectVertex(v):
    x, y, z = v
    canvasX = int((x * d / z * Cw / Vw) + Cw / 2) # +Cw/2 and +Ch/2 are because of PIL meaning (0,0) is top-left. this centers 0,0
    canvasY = int((-y * d / z * Ch / Vh) + Ch / 2) #negative because PIL flips y access
    return (canvasX, canvasY)


def DrawTriangle (P0, P1, P2, color) :
    DrawLine(P0, P1, color);
    DrawLine(P1, P2, color);
    DrawLine(P2, P0, color);

def cube(centerPoint):
    cx,cy,cz = centerPoint

     # The four "front" vertices (z = near face)
    vAf = [cx - 1, cy - 1, cz - 1]
    vBf = [cx - 1, cy + 1, cz - 1]
    vCf = [cx + 1, cy + 1, cz - 1]
    vDf = [cx + 1, cy - 1, cz - 1]

    # The four "back" vertices (z = far face)
    vAb = [cx - 1, cy - 1, cz + 1]
    vBb = [cx - 1, cy + 1, cz + 1]
    vCb = [cx + 1, cy + 1, cz + 1]
    vDb = [cx + 1, cy - 1, cz + 1]

    # Front face (Blue)
    DrawTriangle(ProjectVertex(vAf), ProjectVertex(vBf), ProjectVertex(vCf), BLUE)
    DrawTriangle(ProjectVertex(vAf), ProjectVertex(vCf), ProjectVertex(vDf), RED)

    # Back face (Red)
    DrawTriangle(ProjectVertex(vAb), ProjectVertex(vBb), ProjectVertex(vCb), MAGENTA)
    DrawTriangle(ProjectVertex(vAb), ProjectVertex(vCb), ProjectVertex(vDb), GREEN)

    # Top face
    DrawTriangle(ProjectVertex(vBf), ProjectVertex(vBb), ProjectVertex(vCb), CYAN)
    DrawTriangle(ProjectVertex(vBf), ProjectVertex(vCb), ProjectVertex(vCf), YELLOW)

    # Bottom face
    DrawTriangle(ProjectVertex(vAf), ProjectVertex(vAb), ProjectVertex(vDb), ORANGE)
    DrawTriangle(ProjectVertex(vAf), ProjectVertex(vDb), ProjectVertex(vDf), PURPLE)

    # Left face
    DrawTriangle(ProjectVertex(vAf), ProjectVertex(vAb), ProjectVertex(vBb), PINK)
    DrawTriangle(ProjectVertex(vAf), ProjectVertex(vBb), ProjectVertex(vBf), WHITE)

    # Right face
    DrawTriangle(ProjectVertex(vDf), ProjectVertex(vDb), ProjectVertex(vCb), GRAY)
    DrawTriangle(ProjectVertex(vDf), ProjectVertex(vCb), ProjectVertex(vCf), BROWN)

cube([-2,4,7])
cube([3,3,10])
cube([1,-3,6])
cube([-2,-2,15])


def generate_sphere_vertices(center, radius, stackCount, sectorCount):
    cx, cy, cz = center
    vertices = []

    for i in range(stackCount + 1):
        stack_angle = math.pi / 2 - i * math.pi / stackCount  # from pi/2 to -pi/2
        xy = radius * math.cos(stack_angle)                  # radius of current stack circle
        z = radius * math.sin(stack_angle)

        for j in range(sectorCount + 1):
            sector_angle = j * 2 * math.pi / sectorCount    # 0 to 2pi
            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)
            vertices.append([cx + x, cy + y, cz + z])

    return vertices

def draw_sphere(center, radius=1, stackCount=8, sectorCount=8):
    vertices = generate_sphere_vertices(center, radius, stackCount, sectorCount)

    for i in range(stackCount):
        k1 = i * (sectorCount + 1)        # current stack start index
        k2 = k1 + sectorCount + 1         # next stack start index

        for j in range(sectorCount):
            # skip first stack for first triangle
            if i != 0:
                v1 = vertices[k1]
                v2 = vertices[k2]
                v3 = vertices[k1 + 1]
                DrawTriangle(ProjectVertex(v1), ProjectVertex(v2), ProjectVertex(v3), random.choice(COLORS[1:6]))

            # skip last stack for second triangle
            if i != (stackCount - 1):
                v1 = vertices[k1 + 1]
                v2 = vertices[k2]
                v3 = vertices[k2 + 1]
                DrawTriangle(ProjectVertex(v1), ProjectVertex(v2), ProjectVertex(v3), random.choice(COLORS[1:6]))

            k1 += 1
            k2 += 1

draw_sphere((-3, 0, 5))

img.show()
