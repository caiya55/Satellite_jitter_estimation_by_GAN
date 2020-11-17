
from PIL import Image
import numpy as np

a = Image.open('images//1.png')
img = np.asarray(a)
img = img[:256,:256,:3]
im = Image.fromarray(img)
im.save("images//a1.png")

a = Image.open('images//2.png')
img = np.asarray(a)
img = img[:256,:256,:3]
im = Image.fromarray(img)
im.save("images//a2.png")