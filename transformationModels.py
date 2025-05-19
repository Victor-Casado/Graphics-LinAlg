from PIL import Image
import math
import random

# Constants
Cw, Ch = 500, 500  # Canvas width and height
Vw, Vh = 2.0, 2.0  # Viewport width and height
d = 1.0            # Distance from camera to projection plane

# Colors
COLORS = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (255, 165, 0), (128, 0, 128),
    (255, 105, 180), (255, 255, 255), (128, 128, 128), (139, 69, 19)
]

# Create a new image
img = Image.new('RGB', (Cw, Ch), color='black')
pixels = img.load()

# Useful

def Interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i1 - i0):
        values.append(d)
        d += a
    return values

def DrawLine(P0, P1, color):
    x0, y0 = P0
    x1, y1 = P1
    if abs(x1 - x0) > abs(y1 - y0):
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        ys = Interpolate(x0, y0, x1, y1)
        for x in range(x0, x1):
            y = int(ys[x - x0])
            if 0 <= x < Cw and 0 <= y < Ch:
                pixels[x, y] = color
    else:
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        xs = Interpolate(y0, x0, y1, x1)
        for y in range(y0, y1):
            x = int(xs[y - y0])
            if 0 <= x < Cw and 0 <= y < Ch:
                pixels[x, y] = color

def DrawTriangle(P0, P1, P2, color):
    DrawLine(P0, P1, color)
    DrawLine(P1, P2, color)
    DrawLine(P2, P0, color)

def ProjectVertex(v):
    x, y, z, w = v
    x /= w
    y /= w
    z /= w
    canvasX = int((x * d / z * Cw / Vw) + Cw / 2)
    canvasY = int((-y * d / z * Ch / Vh) + Ch / 2)
    return (canvasX, canvasY)

# Matrices

def identity_matrix():
    return [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]

def translation_matrix(tx, ty, tz):
    return [
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]
    ]

def scaling_matrix(sx, sy, sz):
    return [
        [sx,0,0,0],
        [0,sy,0,0],
        [0,0,sz,0],
        [0,0,0,1]
    ]

def rotation_x_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [1,0,0,0],
        [0,c,-s,0],
        [0,s,c,0],
        [0,0,0,1]
    ]

def rotation_y_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c,0,s,0],
        [0,1,0,0],
        [-s,0,c,0],
        [0,0,0,1]
    ]

def rotation_z_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c,-s,0,0],
        [s,c,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

def matrix_multiply(a, b):
    result = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += a[i][k] * b[k][j]
    return result

def transform_point(m, p):
    return [
        sum(m[i][j] * p[j] for j in range(4))
        for i in range(4)
    ]

# Models

def cube():
    vertices = [
        [-1,-1,-1,1], [-1,1,-1,1], [1,1,-1,1], [1,-1,-1,1],
        [-1,-1,1,1], [-1,1,1,1], [1,1,1,1], [1,-1,1,1],
    ]
    triangles = [
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [1,5,6], [1,6,2], [0,4,7], [0,7,3],
        [0,1,5], [0,5,4], [3,2,6], [3,6,7],
    ]
    return vertices, triangles

def sphere(radius=1.0, stack_count=8, sector_count=8):
    vertices = []
    triangles = []

    for i in range(stack_count + 1):
        stack_angle = math.pi / 2 - i * math.pi / stack_count
        xy = radius * math.cos(stack_angle)
        z = radius * math.sin(stack_angle)

        for j in range(sector_count + 1):
            sector_angle = j * 2 * math.pi / sector_count
            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)
            vertices.append([x, y, z, 1])

    for i in range(stack_count):
        k1 = i * (sector_count + 1)
        k2 = k1 + sector_count + 1

        for j in range(sector_count):
            if i != 0:
                triangles.append([k1 + j, k2 + j, k1 + j + 1])
            if i != (stack_count - 1):
                triangles.append([k1 + j + 1, k2 + j, k2 + j + 1])

    return vertices, triangles

# Render

def render_model(vertices, triangles, model_matrix, view_matrix):
    m = matrix_multiply(view_matrix, model_matrix)
    transformed = [transform_point(m, v) for v in vertices]

    for tri in triangles:
        p0 = ProjectVertex(transformed[tri[0]])
        p1 = ProjectVertex(transformed[tri[1]])
        p2 = ProjectVertex(transformed[tri[2]])
        DrawTriangle(p0, p1, p2, random.choice(COLORS))

def display_model(model_fn, scale=(1,1,1), rotation=(0,0,0), translation=(0,0,0), cameraRotation=(0,0,0), cameraTranslation=(0,0,0)):
    sx, sy, sz = scale
    rx, ry, rz = rotation
    rx *= math.pi/180
    ry *= math.pi/180
    rz *= math.pi/180
    tx, ty, tz = translation

    S = scaling_matrix(sx, sy, sz)
    Rx = rotation_x_matrix(rx)
    Ry = rotation_y_matrix(ry)
    Rz = rotation_z_matrix(rz)
    T = translation_matrix(tx, ty, tz)

    model_matrix = matrix_multiply(T, matrix_multiply(Rz, matrix_multiply(Ry, matrix_multiply(Rx, S))))

    crx, cry, crz = cameraRotation
    crx *= math.pi/180
    cry *= math.pi/180
    crz *= math.pi/180
    ctx, cty, ctz = cameraTranslation

    cRx = rotation_x_matrix(crx)
    cRy = rotation_y_matrix(cry)
    cRz = rotation_z_matrix(crz)
    cT = translation_matrix(ctx, cty, ctz)

    model_matrix = matrix_multiply(cT, matrix_multiply(cRz, matrix_multiply(cRy, cRx)))

    vertices, triangles = model_fn()
    render_model(vertices, triangles, model_matrix, view_matrix)

# Display

display_model(cube, scale=(1,1,1), rotation=(30, 0, 0), translation=(0, 0, 5))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(-2, 0, 6))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(2, 0, 6))
display_model(cube, rotation=(45,60,0), translation = (-2,4,7))

img.show()
