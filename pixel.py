from PIL import Image

# Create a new image (e.g., 200x100 pixels, RGB mode, black background)
img = Image.new('RGB', (200, 100), color='black')

# Access pixel data
pixels = img.load()

# Modify individual pixels (e.g., set pixel at (10, 50) to red)
pixels[50, 50] = (255, 0, 0)

# Display the image (using Pillow's show method or saving to a file)
img.show()
# img.save("output.png")
