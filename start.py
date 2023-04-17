import pygame as pg
import sys
import math
#import taichi as ti

deg45 = math.sqrt(2)/2

flow_list = [[[deg45,deg45],[0,1.0],[0,1.0],[-deg45,deg45]],
             [[1.0,0],[deg45,deg45],[-deg45,deg45],[-1.0,0]],
             [[1.0,0],[deg45,-deg45],[-deg45,-deg45],[-1.0,0]],
             [[deg45,-deg45],[0,-1.0],[0,-1.0],[-deg45,-deg45]]]

pg.init()

color = [(255,255,255),(0,0,0)]
screen = pg.display.set_mode((400,400))  
start  = [200,50]
clock = pg.time.Clock()

while True:
    
    for i in range(0,4):
        for j in range(0,4):
            pg.draw.rect(screen, color[(i+j)%2], (i*100,j*100,100,100))
            pg.draw.line(screen, (0,0,255),(50+i*100,50+j*100),((50+flow_list[i][j][0]*50+i*100),(50-flow_list[i][j][1]*50+j*100)),5)
            
    start[0] += flow_list[int(start[0]/100)][int(start[1]/100)][0]
    start[1] += flow_list[int(start[0]/100)][int(start[1]/100)][1]*-1
    
    print(start)
            
    pg.draw.circle(screen, (255,0,0), (start[0],start[1]), 6)
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
            
    pg.display.set_caption(f'FPS: {clock.get_fps() :.2f}')
    clock.tick(60)
    pg.display.flip()