import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def holdidx_to_coords(value):
    x,y = value
    if x > 32: #kickboard
        x -= 33
        radius = 20
        y_coord = int( x * 30 + 1110)
        x_coord = int( y * 60.0 + int(x == 0) * 30 + 30)
    elif x < 3 or ((x % 2) == 0 and x > 2): #main_holds
        radius = 30
        y_coord = int((min(x,2) + 0.5) * 60.0 + max(0,(x-2)) * 30)
        x_coord = int((y + 1) * 60.0)
    elif x > 2 and (x % 2) == 1: #auxillary holds
        x -= 3
        radius = 15
        y_coord = int(x  * 30 + 180 )
        x_coord = int( y  * 60 + (x % 4) * 30 + 30) 
    else:
        print('bad hold')
    return x_coord, y_coord, radius

def plot_climb(climb_features, thickness=2):
    image = cv2.imread('./figures/full_board_commercial.png')
    fig = plt.figure(figsize=(10, 10))

    for channel, color in zip([3,1,0,2], [(0, 165, 255), (255, 255, 0), (0, 255, 0),  (255, 0, 255)]):
        holds = np.transpose(np.nonzero(climb_features[channel]))
        if len(holds) > 0:
            for x,y,r in np.apply_along_axis(holdidx_to_coords, 1, holds):
                image = cv2.circle(image, (x, y), r, color, thickness)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    return fig
