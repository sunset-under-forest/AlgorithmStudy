import os
import pygame
from pygame.locals import *
import random
import sys
import time
pygame.mixer.pre_init(44100, -16, 2, 2048)  # fix audio delay
pygame.init()

scr_size = (width, height) = (1200, 200)
FPS = 60
gravity = 0.6

black = (0, 0, 0)
white = (255, 255, 255)
background_col = (235, 235, 235)

high_score = 0

screen = pygame.display.set_mode(scr_size)
clock = pygame.time.Clock()
pygame.display.set_caption("T-Rex Rush")

jump_sound = pygame.mixer.Sound('sprites/jump.wav')
die_sound = pygame.mixer.Sound('sprites/die.wav')
checkPoint_sound = pygame.mixer.Sound('sprites/checkPoint.wav')


def load_image(
        name,
        sizex=-1,
        sizey=-1,
        colorkey=None,
):
    fullname = os.path.join('sprites', name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return (image, image.get_rect())


def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex=-1,
        scaley=-1,
        colorkey=None,
):
    fullname = os.path.join('sprites', sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width / nx
    sizey = sheet_rect.height / ny

    for i in range(0, ny):
        for j in range(0, nx):
            rect = pygame.Rect((j * sizex, i * sizey, sizex, sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet, (0, 0), rect)

            if colorkey is not None:
                if colorkey == -1:
                    colorkey = image.get_at((0, 0))
                image.set_colorkey(colorkey, RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image, (scalex, scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites, sprite_rect


def disp_gameOver_msg(retbutton_image, gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width / 2
    retbutton_rect.top = height * 0.52

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width / 2
    gameover_rect.centery = height * 0.35

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)


def extractDigits(number):
    if number > -1:
        digits = []
        i = 0
        while (number / 10 != 0):
            digits.append(number % 10)
            number = int(number / 10)

        digits.append(number % 10)
        for i in range(len(digits), 5):
            digits.append(0)
        digits.reverse()
        return digits

