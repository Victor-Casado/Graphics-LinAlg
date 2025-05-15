from PIL import Image


# Constants
Cw, Ch = 500, 500       # Canvas width and height (pixels)
Vw, Vh = 2.0, 2.0         # Viewport width and height (scene units)
d = 1.0                  # Distance from camera to projection plane

# Colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

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

def ViewportToCanvas(x, y):
    canvasX = int((x * Cw / Vw) + Cw / 2)
    canvasY = int((y * Ch / Vh) + Ch / 2)
    return (canvasX, canvasY)


def ProjectVertex(v):
    x, y, z = v
    return ViewportToCanvas(x * d / z, y * d / z)


# The four "front" vertices
vAf = [-3, -1, 6]
vBf = [-3,  1, 6]
vCf = [-1,  1, 6]
vDf = [-1, -1, 6]

# The four "back" vertices
vAb = [-3, -1, 8]
vBb = [-3,  1, 8]
vCb = [-1,  1, 8]
vDb = [-1, -1, 8]

# The front face
DrawLine(ProjectVertex(vAf), ProjectVertex(vBf), BLUE);
DrawLine(ProjectVertex(vBf), ProjectVertex(vCf), BLUE);
DrawLine(ProjectVertex(vCf), ProjectVertex(vDf), BLUE);
DrawLine(ProjectVertex(vDf), ProjectVertex(vAf), BLUE);

# The back face
DrawLine(ProjectVertex(vAb), ProjectVertex(vBb), RED);
DrawLine(ProjectVertex(vBb), ProjectVertex(vCb), RED);
DrawLine(ProjectVertex(vCb), ProjectVertex(vDb), RED);
DrawLine(ProjectVertex(vDb), ProjectVertex(vAb), RED);

# The front-to-back edges
DrawLine(ProjectVertex(vAf), ProjectVertex(vAb), GREEN);
DrawLine(ProjectVertex(vBf), ProjectVertex(vBb), GREEN);
DrawLine(ProjectVertex(vCf), ProjectVertex(vCb), GREEN);
DrawLine(ProjectVertex(vDf), ProjectVertex(vDb), GREEN);


img.show()
