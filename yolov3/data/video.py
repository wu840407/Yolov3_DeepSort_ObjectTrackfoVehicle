import cv2
import os
name="./car.txt"


if __name__ == '__main__':
    
    file = open(name, 'r')
    line = file.readline()
    shape = cv2.imread(line.split("\n")[0]).shape # delete dimension 3
    size = (shape[1], shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 15, size)
    while line:
    
        img = cv2.imread(line.split("\n")[0])
        out.write(img)
        line = file.readline()
    file.close()
    out.release()