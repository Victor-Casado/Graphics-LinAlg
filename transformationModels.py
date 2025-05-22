from PIL import Image
import math
import random

# canvas is the pixel grid we draw on
# viewport is like a window in 3d space that maps to our canvas
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

# clipping planes define what we can see
# anything outside these planes gets cut off
def compute_planes():
    # get half the viewport size
    hw = Vw / 2
    hh = Vh / 2

    # corners of the viewport rectangle
    top_left     = (-hw,  hh, d)
    top_right    = ( hw,  hh, d)
    bottom_left  = (-hw, -hh, d)
    bottom_right = ( hw, -hh, d)

    origin = (0, 0, 0)

    # each plane is like a wall that cuts off geometry
    # format is (A, B, C, D) where Ax + By + Cz + D = 0
    near_plane  = (0, 0, 1, -d)  # stuff too close gets clipped
    far_plane   = (0, 0, -1, maxDist)  # stuff too far gets clipped

    # the 4 side planes form the edges of what we can see
    left_plane   = plane_from_points(origin, top_left, bottom_left)
    right_plane  = plane_from_points(origin, bottom_right, top_right)
    top_plane    = plane_from_points(origin, top_right, top_left)
    bottom_plane = plane_from_points(origin, bottom_left, bottom_right)

    return [near_plane, far_plane, left_plane, right_plane, top_plane, bottom_plane]

# cross product gives us a vector perpendicular to two other vectors
def cross(u, v):
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]

# make a vector have length 1
def normalize(v):
    mag = sum(c**2 for c in v) ** 0.5
    return [c / mag for c in v]

# make a plane equation from 3 points
def plane_from_points(p1, p2, p3):
    # make two vectors on the plane
    v1 = [p2[i] - p1[i] for i in range(3)]
    v2 = [p3[i] - p1[i] for i in range(3)]

    # cross product gives us the normal (perpendicular) vector
    normal = normalize(cross(v1, v2))
    A, B, C = normal
    # plug in one of the points to get D
    D = -(A*p1[0] + B*p1[1] + C*p1[2])
    return (A, B, C, D)

# interpolate between two values
# this fills in the gaps when drawing lines
def Interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return [d0]
    values = []
    # how much to change each step
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(i1 - i0):
        values.append(d)
        d += a
    return values

# draw a line between two points
def DrawLine(P0, P1, color):
    x0, y0 = P0
    x1, y1 = P1
    # pick the direction that changes more to avoid gaps
    if abs(x1 - x0) > abs(y1 - y0):
        # x changes more so we step through x values
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        # get all the y values for each x
        ys = Interpolate(x0, y0, x1, y1)
        for x in range(x0, x1):
            y = int(ys[x - x0])
            # make sure we dont draw outside the canvas
            if 0 <= x < Cw and 0 <= y < Ch:
                pixels[x, y] = color
    else:
        # y changes more so we step through y values
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        # get all the x values for each y
        xs = Interpolate(y0, x0, y1, x1)
        for y in range(y0, y1):
            x = int(xs[y - y0])
            # make sure we dont draw outside the canvas
            if 0 <= x < Cw and 0 <= y < Ch:
                pixels[x, y] = color

# draw a triangle by drawing its 3 edges
def DrawTriangle(P0, P1, P2, color):
    DrawLine(P0, P1, color)
    DrawLine(P1, P2, color)
    DrawLine(P2, P0, color)

# convert 3d point to 2d screen coordinates
def ProjectVertex(v):
    x, y, z, w = v
    # convert from 4d to 3d coordinates
    x /= w
    y /= w
    z /= w
    # perspective projection - things farther away look smaller
    # divide by z to make distant things smaller
    canvasX = int((x * d / z * Cw / Vw) + Cw / 2)
    canvasY = int((-y * d / z * Ch / Vh) + Ch / 2)  # flip y axis
    return (canvasX, canvasY)

# find where a line crosses a plane
def intersect_line_plane(p1, p2, plane):
    A, B, C, D = plane
    x1, y1, z1, w1 = p1
    x2, y2, z2, w2 = p2

    # direction the line is going
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dw = w2 - w1

    # how far each point is from the plane
    d1 = A * x1 + B * y1 + C * z1 + D
    d2 = A * x2 + B * y2 + C * z2 + D

    # if line is parallel to plane we cant find intersection
    if abs(d1 - d2) < 1e-10:
        return p1

    # find what percentage along the line the intersection is
    t = d1 / (d1 - d2)

    # calculate the intersection point
    x = x1 + t * dx
    y = y1 + t * dy
    z = z1 + t * dz
    w = w1 + t * dw

    return [x, y, z, w]

# check which side of a plane a point is on
def signed_distance_to_plane(plane, point):
    A, B, C, D = plane
    x, y, z, w = point
    return A*x + B*y + C*z + D

# cut a triangle against a clipping plane
def ClipTriangle(triangle, plane, vertices):
    # get the 3 vertices of the triangle
    v0_idx, v1_idx, v2_idx = triangle
    v0, v1, v2 = vertices[v0_idx], vertices[v1_idx], vertices[v2_idx]

    # check which side of the plane each vertex is on
    d0 = signed_distance_to_plane(plane, v0)
    d1 = signed_distance_to_plane(plane, v1)
    d2 = signed_distance_to_plane(plane, v2)

    # small number to handle floating point errors
    eps = 1e-6

    # if all vertices are inside keep the triangle
    if d0 >= -eps and d1 >= -eps and d2 >= -eps:
        return [triangle]

    # if all vertices are outside throw away the triangle
    if d0 < -eps and d1 < -eps and d2 < -eps:
        return []

    # some vertices are inside some are outside
    # figure out which ones
    inside = [(d0 >= -eps, 0), (d1 >= -eps, 1), (d2 >= -eps, 2)]
    inside_verts = [i for is_in, i in inside if is_in]
    outside_verts = [i for is_in, i in inside if not is_in]

    if len(inside_verts) == 1:
        # 1 vertex inside 2 outside
        # result is 1 smaller triangle
        inside_idx = triangle[inside_verts[0]]
        outside_idx1 = triangle[outside_verts[0]]
        outside_idx2 = triangle[outside_verts[1]]

        inside_v = vertices[inside_idx]
        outside_v1 = vertices[outside_idx1]
        outside_v2 = vertices[outside_idx2]

        # find where the edges cross the plane
        int1 = intersect_line_plane(inside_v, outside_v1, plane)
        int2 = intersect_line_plane(inside_v, outside_v2, plane)

        # add the new intersection points to our vertex list
        new_idx1 = len(vertices)
        new_idx2 = len(vertices) + 1
        vertices.extend([int1, int2])

        return [[inside_idx, new_idx1, new_idx2]]

    elif len(inside_verts) == 2:
        # 2 vertices inside 1 outside
        # result is 2 triangles
        inside_idx1 = triangle[inside_verts[0]]
        inside_idx2 = triangle[inside_verts[1]]
        outside_idx = triangle[outside_verts[0]]

        inside_v1 = vertices[inside_idx1]
        inside_v2 = vertices[inside_idx2]
        outside_v = vertices[outside_idx]

        # find where edges cross the plane
        int1 = intersect_line_plane(outside_v, inside_v1, plane)
        int2 = intersect_line_plane(outside_v, inside_v2, plane)

        # add new vertices
        new_idx1 = len(vertices)
        new_idx2 = len(vertices) + 1
        vertices.extend([int1, int2])

        return [[inside_idx1, inside_idx2, new_idx1], [inside_idx2, new_idx2, new_idx1]]

    return []

# clip all triangles against one plane
def ClipTrianglesAgainstPlane(triangles, plane, vertices):
    clipped_triangles = []
    for triangle in triangles:
        clipped = ClipTriangle(triangle, plane, vertices)
        clipped_triangles.extend(clipped)
    return clipped_triangles

# 4x4 identity matrix - no change
def identity_matrix():
    return [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ]

# matrix to move things around
def translation_matrix(tx, ty, tz):
    return [
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]
    ]

# matrix to make things bigger or smaller
def scaling_matrix(sx, sy, sz):
    return [
        [sx,0,0,0],
        [0,sy,0,0],
        [0,0,sz,0],
        [0,0,0,1]
    ]

# matrix to rotate around x axis
def rotation_x_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [1,0,0,0],
        [0,c,-s,0],
        [0,s,c,0],
        [0,0,0,1]
    ]

# matrix to rotate around y axis
def rotation_y_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c,0,s,0],
        [0,1,0,0],
        [-s,0,c,0],
        [0,0,0,1]
    ]

# matrix to rotate around z axis
def rotation_z_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return [
        [c,-s,0,0],
        [s,c,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

# multiply two 4x4 matrices
def matrix_multiply(a, b):
    result = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i][j] += a[i][k] * b[k][j]
    return result

# apply a matrix transformation to a point
def transform_point(m, p):
    return [
        sum(m[i][j] * p[j] for j in range(4))
        for i in range(4)
    ]

# make a cube shape
def cube():
    # 8 vertices of a cube
    vertices = [
        [-1,-1,-1,1], [-1,1,-1,1], [1,1,-1,1], [1,-1,-1,1],
        [-1,-1,1,1], [-1,1,1,1], [1,1,1,1], [1,-1,1,1],
    ]
    # 12 triangles to make 6 faces
    triangles = [
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [1,5,6], [1,6,2], [0,4,7], [0,7,3],
        [0,1,5], [0,5,4], [3,2,6], [3,6,7],
    ]
    return vertices, triangles

# make a sphere shape
def sphere(radius=1.0, stack_count=8, sector_count=8):
    vertices = []
    triangles = []

    # create vertices by going around in circles at different heights
    for i in range(stack_count + 1):
        # angle from top to bottom
        stack_angle = math.pi / 2 - i * math.pi / stack_count
        xy = radius * math.cos(stack_angle)  # radius at this height
        z = radius * math.sin(stack_angle)   # height

        # go around in a circle at this height
        for j in range(sector_count + 1):
            sector_angle = j * 2 * math.pi / sector_count
            x = xy * math.cos(sector_angle)
            y = xy * math.sin(sector_angle)
            vertices.append([x, y, z, 1])

    # connect vertices with triangles
    for i in range(stack_count):
        k1 = i * (sector_count + 1)
        k2 = k1 + sector_count + 1

        for j in range(sector_count):
            # make 2 triangles for each square section
            if i != 0:
                triangles.append([k1 + j, k2 + j, k1 + j + 1])
            if i != (stack_count - 1):
                triangles.append([k1 + j + 1, k2 + j, k2 + j + 1])

    return vertices, triangles

# main rendering function
def render_model(vertices, triangles, model_matrix, view_matrix):
    # transform all vertices
    m = matrix_multiply(view_matrix, model_matrix)
    transformed_vertices = [transform_point(m, v) for v in vertices]

    # start with all triangles
    clipped_triangles = triangles[:]
    clipped_vertices = transformed_vertices[:]

    # clip against each plane
    for plane in planes:
        if not clipped_triangles:
            break
        try:
            clipped_triangles = ClipTrianglesAgainstPlane(clipped_triangles, plane, clipped_vertices)
        except Exception as e:
            print("error")
            continue

    # draw the triangles that are left
    for tri in clipped_triangles:
        try:
            if all(idx < len(clipped_vertices) for idx in tri):
                # convert 3d points to 2d screen coordinates
                p0 = ProjectVertex(clipped_vertices[tri[0]])
                p1 = ProjectVertex(clipped_vertices[tri[1]])
                p2 = ProjectVertex(clipped_vertices[tri[2]])
                # draw the triangle
                DrawTriangle(p0, p1, p2, random.choice(COLORS))
        except Exception as e:
            print("error")
            continue

# helper function to display a model with transformations
def display_model(model_fn, scale=(1,1,1), rotation=(0,0,0), translation=(0,0,0), cameraRotation=(0,0,0), cameraTranslation=(0,0,0)):
    # get scale rotate translate values
    sx, sy, sz = scale
    rx, ry, rz = rotation
    # convert degrees to radians
    rx *= math.pi/180
    ry *= math.pi/180
    rz *= math.pi/180
    tx, ty, tz = translation

    # make transformation matrices
    S = scaling_matrix(sx, sy, sz)
    Rx = rotation_x_matrix(rx)
    Ry = rotation_y_matrix(ry)
    Rz = rotation_z_matrix(rz)
    T = translation_matrix(tx, ty, tz)

    # combine transformations - order matters
    model_matrix = matrix_multiply(T, matrix_multiply(Rz, matrix_multiply(Ry, matrix_multiply(Rx, S))))

    # camera transformations
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

    # get the model and render it
    vertices, triangles = model_fn()
    render_model(vertices, triangles, model_matrix, view_matrix)

# set up the clipping planes
planes = compute_planes()

# render some objects
display_model(cube, scale=(1,1,1), rotation=(30, 0, 0), translation=(5, 0, 5))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(-2, 0, 6))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(2, 0, 6))
display_model(sphere, scale=(2,2,2), translation = (0,3,10))
display_model(cube, translation= (0,0,7))
img.show()
