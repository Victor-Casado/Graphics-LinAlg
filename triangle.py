from PIL import Image

img = Image.new('RGB', (1000, 500), color='black')
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

def DrawWireframeTriangle (P0, P1, P2, color) :
    DrawLine(P0, P1, color);
    DrawLine(P1, P2, color);
    DrawLine(P2, P0, color);

DrawWireframeTriangle([50,50], [100,100], [200, 100], (255, 0, 0))
img.show()
