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


def compute_planes():
    hw = Vw / 2
    hh = Vh / 2

    # Viewport corners
    top_left     = (-hw,  hh, d)
    top_right    = ( hw,  hh, d)
    bottom_left  = (-hw, -hh, d)
    bottom_right = ( hw, -hh, d)

    origin = (0, 0, 0)

    # Planes: all normals should point *inward* (into frustum)
    near_plane  = (0, 0, 1, -d)  # z >= d (near plane at z=d)
    far_plane   = (0, 0, -1, maxDist)  # z <= maxDist

    left_plane   = plane_from_points(origin, top_left, bottom_left)
    right_plane  = plane_from_points(origin, bottom_right, top_right)
    top_plane    = plane_from_points(origin, top_right, top_left)
    bottom_plane = plane_from_points(origin, bottom_left, bottom_right)

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

    if abs(d1 - d2) < 1e-10:  # Line is parallel to plane
        return p1

    # percent along line computed to get to plane
    t = d1 / (d1 - d2)

    # move along the line
    x = x1 + t * dx
    y = y1 + t * dy
    z = z1 + t * dz
    w = w1 + t * dw

    return [x, y, z, w]


def signed_distance_to_plane(plane, point):
    A, B, C, D = plane
    x, y, z, w = point
    return A*x + B*y + C*z + D


def ClipTriangle(triangle, plane, vertices):
    v0_idx, v1_idx, v2_idx = triangle
    v0, v1, v2 = vertices[v0_idx], vertices[v1_idx], vertices[v2_idx]

    d0 = signed_distance_to_plane(plane, v0)
    d1 = signed_distance_to_plane(plane, v1)
    d2 = signed_distance_to_plane(plane, v2)

    # Use small epsilon for floating point errors
    eps = 1e-6

    # All vertices are inside
    if d0 >= -eps and d1 >= -eps and d2 >= -eps:
        return [triangle]

    # All vertices are outside
    if d0 < -eps and d1 < -eps and d2 < -eps:
        return []

    inside = [(d0 >= -eps, 0), (d1 >= -eps, 1), (d2 >= -eps, 2)]
    inside_verts = [i for is_in, i in inside if is_in]
    outside_verts = [i for is_in, i in inside if not is_in]

    if len(inside_verts) == 1:
        inside_idx = triangle[inside_verts[0]]
        outside_idx1 = triangle[outside_verts[0]]
        outside_idx2 = triangle[outside_verts[1]]

        inside_v = vertices[inside_idx]
        outside_v1 = vertices[outside_idx1]
        outside_v2 = vertices[outside_idx2]

        # Find intersections
        int1 = intersect_line_plane(inside_v, outside_v1, plane)
        int2 = intersect_line_plane(inside_v, outside_v2, plane)

        # Add new vertices
        new_idx1 = len(vertices)
        new_idx2 = len(vertices) + 1
        vertices.extend([int1, int2])

        return [[inside_idx, new_idx1, new_idx2]]

    elif len(inside_verts) == 2:
        inside_idx1 = triangle[inside_verts[0]]
        inside_idx2 = triangle[inside_verts[1]]
        outside_idx = triangle[outside_verts[0]]

        inside_v1 = vertices[inside_idx1]
        inside_v2 = vertices[inside_idx2]
        outside_v = vertices[outside_idx]

        # Find intersections
        int1 = intersect_line_plane(outside_v, inside_v1, plane)
        int2 = intersect_line_plane(outside_v, inside_v2, plane)

        # Add new vertices
        new_idx1 = len(vertices)
        new_idx2 = len(vertices) + 1
        vertices.extend([int1, int2])

        return [[inside_idx1, inside_idx2, new_idx1], [inside_idx2, new_idx2, new_idx1]]

    return []


def ClipTrianglesAgainstPlane(triangles, plane, vertices):
    clipped_triangles = []
    for triangle in triangles:
        clipped = ClipTriangle(triangle, plane, vertices)
        clipped_triangles.extend(clipped)
    return clipped_triangles


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
    # Transform vertices
    m = matrix_multiply(view_matrix, model_matrix)
    transformed_vertices = [transform_point(m, v) for v in vertices]

    # Create working copies for clipping
    clipped_triangles = triangles[:]
    clipped_vertices = transformed_vertices[:]

    for plane in planes:
        if not clipped_triangles:
            break
        try:
            clipped_triangles = ClipTrianglesAgainstPlane(clipped_triangles, plane, clipped_vertices)
        except Exception as e:
            print("error")
            continue

    # Project and draw triangles
    for tri in clipped_triangles:
        try:
            if all(idx < len(clipped_vertices) for idx in tri):
                p0 = ProjectVertex(clipped_vertices[tri[0]])
                p1 = ProjectVertex(clipped_vertices[tri[1]])
                p2 = ProjectVertex(clipped_vertices[tri[2]])
                DrawTriangle(p0, p1, p2, random.choice(COLORS))
        except Exception as e:
            print("error")
            continue

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
    render_model(vertices, triangles, model_matrix, view_matrix)

planes = compute_planes()


# Display
display_model(cube, scale=(1,1,1), rotation=(30, 0, 0), translation=(5, 0, 5))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(-2, 0, 6))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(2, 0, 6))
display_model(sphere, scale=(2,2,2), translation = (0,0,1))
display_model(cube, translation= (0,0,7))
img.show()
