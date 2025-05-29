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
    (255, 105, 180), (128, 128, 128), (139, 69, 19)
]

# Lighting parameters
LIGHT_POS = [5, 5, 5]  # position of light source
LIGHT_COLOR = [1.0, 1.0, 1.0]  # white light
AMBIENT_STRENGTH = 0.2
DIFFUSE_STRENGTH = 0.8
SPECULAR_STRENGTH = 0.5
SHININESS = 32

# Camera position (for specular calculations)
CAMERA_POS = [0, 0, 0]

# Create a new image and depth buffer
img = Image.new('RGB', (Cw, Ch), color='black')
pixels = img.load()

# Depth buffer - initialize to 0 (representing 1/infinity)
depth_buffer = [[0.0 for _ in range(Ch)] for _ in range(Cw)]

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
    if mag == 0:
        return [0, 0, 0]
    return [c / mag for c in v]

# vector subtraction
def subtract_vectors(v1, v2):
    return [v1[i] - v2[i] for i in range(3)]

# vector addition
def add_vectors(v1, v2):
    return [v1[i] + v2[i] for i in range(3)]

# dot product
def dot_product(v1, v2):
    return sum(v1[i] * v2[i] for i in range(3))

# reflect vector around normal
def reflect_vector(v, n):
    dot_vn = dot_product(v, n)
    return [v[i] - 2 * dot_vn * n[i] for i in range(3)]

# calculate normal vector for a triangle
def calculate_triangle_normal(v0, v1, v2):
    # convert from homogeneous coordinates
    p0 = [v0[0]/v0[3], v0[1]/v0[3], v0[2]/v0[3]]
    p1 = [v1[0]/v1[3], v1[1]/v1[3], v1[2]/v1[3]]
    p2 = [v2[0]/v2[3], v2[1]/v2[3], v2[2]/v2[3]]

    # two edges of the triangle
    edge1 = subtract_vectors(p1, p0)
    edge2 = subtract_vectors(p2, p0)

    # cross product gives normal
    normal = cross(edge1, edge2)
    return normalize(normal)

# compute phong shading for a point
def compute_phong_shading(point, normal, base_color):
    # convert point from homogeneous coordinates
    world_pos = [point[0]/point[3], point[1]/point[3], point[2]/point[3]]

    # vector from point to light
    light_dir = normalize(subtract_vectors(LIGHT_POS, world_pos))

    # vector from point to camera
    view_dir = normalize(subtract_vectors(CAMERA_POS, world_pos))

    # ambient component
    ambient = [AMBIENT_STRENGTH * base_color[i] / 255.0 for i in range(3)]

    # diffuse component
    diff = max(0, dot_product(normal, light_dir))
    diffuse = [DIFFUSE_STRENGTH * diff * LIGHT_COLOR[i] * base_color[i] / 255.0 for i in range(3)]

    # specular component
    reflect_dir = reflect_vector([-light_dir[i] for i in range(3)], normal)
    spec = max(0, dot_product(view_dir, reflect_dir)) ** SHININESS
    specular = [SPECULAR_STRENGTH * spec * LIGHT_COLOR[i] for i in range(3)]

    # combine all components
    final_color = [
        min(255, int(255 * (ambient[i] + diffuse[i] + specular[i])))
        for i in range(3)
    ]

    return tuple(final_color)

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

# interpolate colors component-wise
def InterpolateColor(i0, c0, i1, c1):
    if i0 == i1:
        return [c0]
    colors = []
    # interpolate each color component separately
    r_vals = Interpolate(i0, c0[0], i1, c1[0])
    g_vals = Interpolate(i0, c0[1], i1, c1[1])
    b_vals = Interpolate(i0, c0[2], i1, c1[2])

    for i in range(len(r_vals)):
        colors.append((int(r_vals[i]), int(g_vals[i]), int(b_vals[i])))
    return colors

# draw a line between two points (with depth testing and color interpolation)
def DrawLine(P0, P1, z0, z1, color0, color1):
    x0, y0 = P0
    x1, y1 = P1

    # Convert Z values to 1/Z for proper interpolation
    inv_z0 = 1.0 / z0 if z0 != 0 else 0
    inv_z1 = 1.0 / z1 if z1 != 0 else 0

    # pick the direction that changes more to avoid gaps
    if abs(x1 - x0) > abs(y1 - y0):
        # x changes more so we step through x values
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
            inv_z0, inv_z1 = inv_z1, inv_z0
            color0, color1 = color1, color0
        # get all the y values, 1/z values, and colors for each x
        ys = Interpolate(x0, y0, x1, y1)
        inv_zs = Interpolate(x0, inv_z0, x1, inv_z1)
        colors = InterpolateColor(x0, color0, x1, color1)
        for x in range(x0, x1):
            if x - x0 >= len(ys) or x - x0 >= len(inv_zs) or x - x0 >= len(colors):
                continue
            y = int(ys[x - x0])
            inv_z = inv_zs[x - x0]
            color = colors[x - x0]
            # make sure we dont draw outside the canvas
            if 0 <= x < Cw and 0 <= y < Ch:
                # Depth test: keep pixel if it's closer (higher 1/z value)
                if inv_z > depth_buffer[x][y]:
                    depth_buffer[x][y] = inv_z
                    pixels[x, y] = color
    else:
        # y changes more so we step through y values
        if y0 > y1:
            x0, y0, x1, y1 = x1, y1, x0, y0
            inv_z0, inv_z1 = inv_z1, inv_z0
            color0, color1 = color1, color0
        # get all the x values, 1/z values, and colors for each y
        xs = Interpolate(y0, x0, y1, x1)
        inv_zs = Interpolate(y0, inv_z0, y1, inv_z1)
        colors = InterpolateColor(y0, color0, y1, color1)
        for y in range(y0, y1):
            if y - y0 >= len(xs) or y - y0 >= len(inv_zs) or y - y0 >= len(colors):
                continue
            x = int(xs[y - y0])
            inv_z = inv_zs[y - y0]
            color = colors[y - y0]
            # make sure we dont draw outside the canvas
            if 0 <= x < Cw and 0 <= y < Ch:
                # Depth test: keep pixel if it's closer (higher 1/z value)
                if inv_z > depth_buffer[x][y]:
                    depth_buffer[x][y] = inv_z
                    pixels[x, y] = color

def DrawFilledTriangle(P0, P1, P2, z0, z1, z2, color0, color1, color2):
    # Sort points by y-coordinate (ascending), keeping z values and colors aligned
    points_with_data = sorted([(P0, z0, color0), (P1, z1, color1), (P2, z2, color2)], key=lambda item: item[0][1])
    (x0, y0), z0, color0 = points_with_data[0]
    (x1, y1), z1, color1 = points_with_data[1]
    (x2, y2), z2, color2 = points_with_data[2]

    # Convert to int for safe range use
    y0, y1, y2 = int(y0), int(y1), int(y2)

    # Convert Z values to 1/Z for proper interpolation
    inv_z0 = 1.0 / z0 if z0 != 0 else 0
    inv_z1 = 1.0 / z1 if z1 != 0 else 0
    inv_z2 = 1.0 / z2 if z2 != 0 else 0

    # Interpolate X values, 1/Z values, and colors along edges
    x01 = Interpolate(y0, x0, y1, x1)
    x12 = Interpolate(y1, x1, y2, x2)
    x02 = Interpolate(y0, x0, y2, x2)

    inv_z01 = Interpolate(y0, inv_z0, y1, inv_z1)
    inv_z12 = Interpolate(y1, inv_z1, y2, inv_z2)
    inv_z02 = Interpolate(y0, inv_z0, y2, inv_z2)

    color01 = InterpolateColor(y0, color0, y1, color1)
    color12 = InterpolateColor(y1, color1, y2, color2)
    color02 = InterpolateColor(y0, color0, y2, color2)

    x012 = x01[:-1] + x12  # avoid duplicating y1 row
    inv_z012 = inv_z01[:-1] + inv_z12
    color012 = color01[:-1] + color12

    if len(x012) != (y2 - y0):
        x012 = x012[:y2 - y0]
        inv_z012 = inv_z012[:y2 - y0]
        color012 = color012[:y2 - y0]

    if len(x02) != (y2 - y0):
        x02 = x02[:y2 - y0]
        inv_z02 = inv_z02[:y2 - y0]
        color02 = color02[:y2 - y0]

    # Determine which side is left or right
    mid = len(x02) // 2
    if mid < len(x02) and mid < len(x012) and x02[mid] < x012[mid]:
        x_left, x_right = x02, x012
        inv_z_left, inv_z_right = inv_z02, inv_z012
        color_left, color_right = color02, color012
    else:
        x_left, x_right = x012, x02
        inv_z_left, inv_z_right = inv_z012, inv_z02
        color_left, color_right = color012, color02

    # Draw horizontal lines with depth testing and color interpolation
    for y in range(y0, y2):
        y_idx = y - y0
        if y_idx >= len(x_left) or y_idx >= len(x_right) or y_idx >= len(color_left) or y_idx >= len(color_right):
            continue
        xl = int(x_left[y_idx])
        xr = int(x_right[y_idx])
        inv_z_l = inv_z_left[y_idx]
        inv_z_r = inv_z_right[y_idx]
        color_l = color_left[y_idx]
        color_r = color_right[y_idx]

        if xl > xr:
            xl, xr = xr, xl
            inv_z_l, inv_z_r = inv_z_r, inv_z_l
            color_l, color_r = color_r, color_l

        # Interpolate 1/Z and color across the horizontal line
        if xl != xr:
            inv_zs = Interpolate(xl, inv_z_l, xr, inv_z_r)
            colors = InterpolateColor(xl, color_l, xr, color_r)
        else:
            inv_zs = [inv_z_l]
            colors = [color_l]

        for x in range(xl, xr):
            x_idx = x - xl
            if x_idx >= len(inv_zs) or x_idx >= len(colors):
                continue
            inv_z = inv_zs[x_idx]
            color = colors[x_idx]
            if 0 <= x < Cw and 0 <= y < Ch:
                # Depth test: keep pixel if it's closer (higher 1/z value)
                if inv_z > depth_buffer[x][y]:
                    depth_buffer[x][y] = inv_z
                    pixels[x, y] = color

# draw a triangle by drawing its 3 edges or filled
def DrawTriangle(P0, P1, P2, z0, z1, z2, color0, color1, color2):
    # Uncomment next 3 lines to draw wireframe
     DrawLine(P0, P1, z0, z1, color0, color1)
     DrawLine(P1, P2, z1, z2, color1, color2)
     DrawLine(P2, P0, z2, z0, color2, color0)

    # Draw filled triangle with phong shading
     DrawFilledTriangle(P0, P1, P2, z0, z1, z2, color0, color1, color2)

# convert 3d point to 2d screen coordinates and return z value
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
    return (canvasX, canvasY, z)  # return z value for depth testing

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
def render_model(vertices, triangles, model_matrix, view_matrix, base_color):
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
        clipped_triangles = ClipTrianglesAgainstPlane(clipped_triangles, plane, clipped_vertices)

    # draw the triangles that are left
    for tri in clipped_triangles:
        if all(idx < len(clipped_vertices) for idx in tri):
            # get the 3 vertices
            v0, v1, v2 = clipped_vertices[tri[0]], clipped_vertices[tri[1]], clipped_vertices[tri[2]]

            # calculate triangle normal for lighting
            normal = calculate_triangle_normal(v0, v1, v2)

            # compute phong shading for each vertex
            color0 = compute_phong_shading(v0, normal, base_color)
            color1 = compute_phong_shading(v1, normal, base_color)
            color2 = compute_phong_shading(v2, normal, base_color)

            # convert 3d points to 2d screen coordinates and get z values
            projected = [ProjectVertex(clipped_vertices[idx]) for idx in tri]
            p0, z0 = (projected[0][0], projected[0][1]), projected[0][2]
            p1, z1 = (projected[1][0], projected[1][1]), projected[1][2]
            p2, z2 = (projected[2][0], projected[2][1]), projected[2][2]

            # draw the triangle with per-vertex colors
            DrawTriangle(p0, p1, p2, z0, z1, z2, color0, color1, color2)

# helper function to display a model with transformations
def display_model(model_fn, scale=(1,1,1), rotation=(0,0,0), translation=(0,0,0), cameraRotation=(0,0,0), cameraTranslation=(0,0,0), color=None):
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

    # use provided color or pick a random one
    if color is None:
        color = random.choice(COLORS)

    render_model(vertices, triangles, model_matrix, view_matrix, color)

# Function to clear the depth buffer for new frames
def clear_depth_buffer():
    global depth_buffer
    depth_buffer = [[0.0 for _ in range(Ch)] for _ in range(Cw)]

# set up the clipping planes
planes = compute_planes()

# Clear depth buffer before rendering
clear_depth_buffer()

# render some objects with phong shading
display_model(cube, rotation=(20, 40, 60), scale=(1,1,1), translation=(1, 0, 4), color=(255, 0, 0))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(-2, 0, 6), color=(0, 255, 0))
display_model(sphere, scale=(0.6,0.6,0.6), translation=(5, 0, 6), color=(0, 0, 255))
display_model(sphere, scale=(2,2,2), translation=(0,3,10), color=(255, 255, 0))
display_model(cube, rotation=(50, 0, 0), translation=(-2,0,7), color=(255, 0, 255))

img.show()
