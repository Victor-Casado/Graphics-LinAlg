from PIL import Image

# Create a new image (e.g., 100x100 pixels, RGB mode, black background)
img = Image.new('RGB', (100, 100), color='black')

# Access pixel data
pixels = img.load()

p0 = [20,3] #x, y
p1 = [50,50]

def DrawLine(p0,p1,color):
    m = (p1[1]-p0[1])/(p1[0]-p0[0]) #delta y over delta x
    b = p0[1] - m * p0[0]
    step = 1
    if (p0[0] > p1[0]):
        step = -1
    for x in range(p0[0], p1[0], step):
        y = m*x + b
        pixels[x, y] = color

DrawLine(p0, p1, (255,0,0))

# Display the image (using Pillow's show method or saving to a file)
img.show()
# img.save("output.png")
