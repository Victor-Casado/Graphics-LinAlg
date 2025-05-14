from PIL import Image

img = Image.new('RGB', (1000, 500), color='black')
pixels = img.load()

def Interpolate(i0, d0, i1, d1):
    if i0 == i1:
        return [d0]
    values = []
    a = (d1 - d0) / (i1 - i0)
    d = d0
    for i in range(int(i1) - int(i0) + 1):
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

def DrawShadedTriangle (P0, P1, P2, color):

    points = sorted([P0, P1, P2], key=lambda p: p[1])
    P0, P1, P2 = points
        #y0 <= y1 <= y2

    x0, y0, h0 = P0
    x1, y1, h1 = P1
    x2, y2, h2 = P2

    x01 = Interpolate(y0, x0, y1, x1) #x values between 0 and 1
    h01 = Interpolate(y0, h0, y1, h1)

    x12 = Interpolate(y1, x1, y2, x2) #x values between 1 and 2
    h12 = Interpolate(y1, h1, y2, h2)

    x02 = Interpolate(y0, x0, y2, x2) #x values between 2 and 0
    h02 = Interpolate(y0, h0, y2, h2)

    x012 = []
    x012.extend(x01)
    x012.extend(x12) #x values throughout

    h012 = []
    h012.extend(h01)
    h012.extend(h12)

    #print(len(x02))
    #print(len(x012))
    #print(y1-y0)
    middle = y1-y0 #determine left and right
    if x02[middle] < x012[middle]:
        x_left = x02
        h_left = h02

        x_right = x012
        h_right = h012
    else:
        x_left = x012
        h_left = h012

        x_right = x02
        h_right = h02

    for y in range(y0, y2):
        leftside = x_left[y-y0]
        rightside = x_right[y-y0]

        shadedsegment = Interpolate(leftside, h_left[y-y0], rightside, h_right[y-y0])

        for x in range(int(leftside), int(rightside)):
            newcolor = (int(round(color[0] * shadedsegment[int(x-leftside)])), int(round(color[1] * shadedsegment[int(x-leftside)])), int(round(color[2] * shadedsegment[int(x-leftside)])))
            pixels[x,y] = newcolor


DrawShadedTriangle([150,150, 0], [200,500, 1], [800, 200, .5], (0, 255, 255))
img.show()
