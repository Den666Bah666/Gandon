from PIL import Image
from dataclasses import dataclass
from sys import argv, stdout
import os

@dataclass
class Color:
    rgb: tuple
    char: str

@dataclass
class Progress:
    total: int = 0
    completed: int = 0
    
    @property
    def perc(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100
    
    def process(self):
        pidr = self.completed + 1
        if pidr < self.total:
            self.completed = pidr

class Nfp:
    white = Color((240, 240, 240), '0')
    orange = Color((242, 178, 51), '1')
    magenta = Color((229, 127, 216), '2')
    light_blue = Color((153, 178, 242), '3')
    yellow = Color((222, 222, 108), '4')
    lime = Color((127, 204, 25), '5')
    pink = Color((242, 178, 204), '6')
    gray = Color((76, 76, 76), '7')
    light_gray = Color((153, 153, 153), '8')
    cyan = Color((76, 153, 178), '9')
    purple = Color((178, 102, 229), 'a')
    blue = Color((51, 102, 204), 'b')
    brown = Color((127, 102, 76), 'c')
    green = Color((87, 166, 78), 'd')
    red = Color((204, 76, 76), 'e')
    black = Color((17, 17, 17), 'f')

def find_closest_color(pixel, palette):
    min_distance = float('inf')
    best_char = '0'
    for color in palette:
        distance = sum((c1 - c2)**2 for c1, c2 in zip(pixel, color.rgb))
        if distance < min_distance:
            min_distance = distance
            best_char = color.char
    return best_char

image = Image.open(f"{argv[1]}.png", "r").convert("RGB")
output = ""
image = image.resize((121, 81)) # 121 81
nfp_colors = [color for name, color in vars(Nfp).items() if isinstance(color, Color)]

progress = Progress(total=image.width * image.height)

os.system("cls")
for y in range(image.height):
    for x in range(image.width):
        pixel_color = image.getpixel((x, y))

        output += find_closest_color(pixel_color, nfp_colors)

        progress.process()
        stdout.write(f"\033[1;1H{progress.perc:.2f}%")
    output += "\n" if y < image.height - 1 else ''

with open(f"{argv[1]}.txt", "w") as file:
    file.write(output)