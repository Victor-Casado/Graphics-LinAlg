from PIL import Image
import math
import random

# Constants
Cw, Ch = 500, 500  # Canvas width and height
Vw, Vh = 2.0, 2.0  # Viewport width and height
d = 1.0            # Distance from camera to projection plane
maxDist = 1000 # Distance at which stuff is no longer rendered
minDist = 0
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

def compute_frustum_planes():
    # Viewport half-dimensions
    hw = Vw / 2
    hh = Vh / 2

    # Viewport corners
    top_left     = (-hw,  hh, d)
    top_right    = ( hw,  hh, d)
    bottom_left  = (-hw, -hh, d)
    bottom_right = ( hw, -hh, d)

    origin = (0, 0, 0)

    # Planes: all normals should point *inward* (into frustum)
    near_plane  = (0, 0, -1, 0)  # z = 0
    far_plane   = (0, 0, 1, -maxDist)

    left_plane   = plane_from_points(origin, bottom_left, bottom_right)
    right_plane  = plane_from_points(origin, top_right, bottom_right)
    top_plane    = plane_from_points(origin, top_right, top_left)
    bottom_plane = plane_from_points(origin, top_left, bottom_left)

    return [near_plane, far_plane, left_plane, right_plane, top_plane, bottom_plane]



def cross(u, v):
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]

def normalize(v):
    mag = sum(c**2 for c in v) ** 0.5
    return [c / mag for c in v]

def plane_from_points(p1, p2, p3):
    # Two vectors on the plane
    v1 = [p2[i] - p1[i] for i in range(3)]
    v2 = [p3[i] - p1[i] for i in range(3)]

    normal = normalize(cross(v1, v2))
    A, B, C = normal
    D = -(A*p1[0] + B*p1[1] + C*p1[2])
    return (A, B, C, D)


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


def intersect_line_plane(p1, p2, plane):
    A, B, C, D = plane
    x1, y1, z1, w1 = p1
    x2, y2, z2, w2 = p2

    # line
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dw = w2 - w1

    # Compute the distances
    d1 = A * x1 + B * y1 + C * z1 + D
    d2 = A * x2 + B * y2 + C * z2 + D

    if d1 == d2:
        return p1  # Line is parallel and on plane

    # percent along line computed to get to plane
    t = d1 / (d1 - d2)

    # move along the line
    x = x1 + t * dx
    y = y1 + t * dy
    z = z1 + t * dz
    w = w1 + t * dw

    return [x, y, z, w]


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

def render_model(vertices, triangles, model_matrix, view_matrix, center):
    #print(center)
    m = matrix_multiply(view_matrix, model_matrix)
    transformed = [transform_point(m, v) for v in vertices]

    largestRad = 0
    for v in transformed:
        if math.dist(v, center) > largestRad:
            largestRad = math.dist(v, center)



            #Culling
    for plane in planes:
        culled = cull_model(center, largestRad, plane)
        #print(culled)
        if culled == "Partially Inside":
            index = 0
            #print(triangles)
            for index in range(len(triangles)):
                #print(index)
                if index >= len(triangles):
                    break

                #print("(" + str(transformed[triangles[index][0]][0]) + "," + str(transformed[triangles[index][0]][1]) + "," + str(transformed[triangles[index][0]][2]) + ")")
                culled0 = cull_model(transformed[triangles[index][0]], 0)

                #print("(" + str(transformed[triangles[index][1]][0]) + "," + str(transformed[triangles[index][1]][1]) + "," + str(transformed[triangles[index][1]][2]) + ")")
                culled1 = cull_model(transformed[triangles[index][1]], 0)

                #print("(" + str(transformed[triangles[index][2]][0]) + "," + str(transformed[triangles[index][2]][1]) + "," + str(transformed[triangles[index][2]][2]) + ")")
                culled2 = cull_model(transformed[triangles[index][2]], 0)

                if culled0 and culled1 and culled2:
                    #print("outside")
                    triangles.pop(index)
                    index -= 1
                else:
                    p0i = triangles[index][0]
                    p1i = triangles[index][1]
                    p2i = triangles[index][2]
                    if (culled0 and culled1) and (not culled2):
                        pass
                    if (culled0 and culled2) and (not culled1):
                        pass
                    if (culled1 and culled2) and (not culled0):
                        pass
                    if ((not culled0) and (not culled1)) and culled2:
                        pass
                        '''
                        let A be the vertex with a positive distance
                        compute B' = Intersection(AB, plane)
                        compute C' = Intersection(AC, plane)
                        return [Triangle(A, B', C')]
                        '''
                    if ((not culled0) and (not culled2)) and culled1:
                        pass
                    if ((not culled1) and (not culled2)) and culled0:
                        pass
                    #intersect_line_plane()

            #print("new triangle\n")
    else:
        if culled:
            return False

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

    view_matrix = matrix_multiply(cT, matrix_multiply(cRz, matrix_multiply(cRy, cRx)))

    vertices, triangles = model_fn()

    center = (translation[0] - cameraTranslation[0], translation[1] - cameraTranslation[1], translation[2] - cameraTranslation[2], 1)
    #print(center)
    render_model(vertices, triangles, model_matrix, view_matrix, center)

# Culling

def signed_distance_to_plane(plane, point):
    A, B, C, D = plane
    x, y, z, w = point
    dist = A*x + B*y + C*z + D
    if math.isclose(0, dist): #floating point
        return 0
    return dist

def cull_model(center, radius, plane):
    distance = signed_distance_to_plane(plane, center)
    sign = signed_distance_to_plane(plane, (0,0,1,1))
    if not abs(distance) < radius:
        if (sign > 0 and distance < 0) or (sign < 0 and distance > 0):
            return True
    else:
        if radius != 0: #because of reutilization for points
            return "Partially Inside"
    return False

# Display
planes = compute_frustum_planes()

display_model(cube, scale=(1,1,1), rotation=(30, 0, 0), translation=(5, 0, 5))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(-2, 0, 6))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(2, 0, 6))
display_model(cube, rotation=(45,60,0), translation = (-2,4,7))
display_model(cube, translation= (0,0,7))
img.show()
