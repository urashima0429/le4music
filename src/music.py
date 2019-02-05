import numpy as np
import math

A4 = 440.0
keys = {
    'C' : -9,
    'Db': -8,
    'D' : -7,
    'Eb': -6,
    'E' : -5,
    'F' : -4,
    'Gb': -3,
    'G' : -2,
    'Ab': -1,
    'A' :  0,
    'Bb':  1,
    'B' :  2,
}

def note_to_Hz(note):
    splited = list(note)
    height = int(splited[-1]) - 4
    if len(splited) == 2:
        tone = keys[splited[0]]
    else:
        tone = keys[splited[0] + splited[1]]
    return A4 * (2 ** (height + tone / 12))

def Hz_to_note(Hz):
    hoge = math.log2(Hz / A4)
    height = int(hoge)
    tone = int(12 * (hoge-height))
    for i in keys:
        if keys[i] == tone:
            return i + str(height + 4)
    return ""

print(note_to_Hz('A4'))