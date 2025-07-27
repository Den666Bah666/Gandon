from PIL import Image
from dataclasses import dataclass
from sys import argv, stdout
import os
import cv2
from moviepy import VideoFileClip
import subprocess
import numpy as np
import soundfile as sf

# argv[1]: filename
# argv[2]: frame_step

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

def cv2_to_pil(cv2_frame):
    """Конвертирует OpenCV frame в PIL Image с проверкой типа"""
    if cv2_frame is None:
        raise ValueError("Пустой кадр")
    
    # Если изображение grayscale (1 канал)
    if len(cv2_frame.shape) == 2:
        return Image.fromarray(cv2_frame)
    
    # Если цветное изображение (3 канала)
    return Image.fromarray(cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB))

def extract_audio(video_path: str, output_audio_path: str) -> None:
    """Extract audio from video file and save as WAV format using moviepy."""
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio.write_audiofile(
            output_audio_path, 
            codec='pcm_s16le',
            ffmpeg_params=['-ar', '48000']
        )

def convert_to_dfpwm(input_wav_path: str, output_dfpwm_path: str) -> None:
    """Convert WAV to DFPWM using soundfile for WAV reading and ffmpeg for conversion."""
    # Читаем WAV файл с помощью soundfile
    data, samplerate = sf.read(input_wav_path, dtype='int16')
    
    # Конвертируем в моно если нужно
    if len(data.shape) > 1:
        data = np.mean(data, axis=1).astype('int16')
    
    # Ресемплируем до 24000 Гц если нужно
    # if samplerate != 24000:
    #     from scipy import signal
    #     samples = int(len(data) * 24000 / samplerate)
    #     data = signal.resample(data, samples).astype('int16')
    
    # Конвертируем через ffmpeg
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 's16le',
        '-ar', '48000',
        '-ac', '1',
        '-i', '-',
        '-acodec', 'dfpwm',
        output_dfpwm_path
    ]
    
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    process.communicate(input=data.tobytes())
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed with code {process.returncode}")

video = cv2.VideoCapture(f"{argv[1]}.mp4")
frames = []
i = 0
_i = 0

if not os.path.exists("videos"):
    os.mkdir("videos")
elif not os.path.exists(f"videos/{argv[1]}"):
    os.mkdir(f"videos/{argv[1]}")

while True:
    ret, frame = video.read()

    if not ret:
        break

    if int(argv[2]) <= 0:
        c = True
    else:
        c = (i + 1) % int(argv[2]) == 0

    if c or i <= 0:
        image = cv2_to_pil(frame)
        output = ""
        image = image.resize((121, 81))
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

        with open(f"videos/{argv[1]}/{_i}.txt", "w") as file:
            file.write(output)

        _i += 1

    i += 1

video.release()

extract_audio(f"{argv[1]}.mp4", f"{argv[1]}.wav")
convert_to_dfpwm(f"{argv[1]}.wav", f"videos/{argv[1]}/audio.dfpwm")