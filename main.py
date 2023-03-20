import pygame as pg
import sys
import math
import numpy as np
import taichi as ti
from taichi_glsl import vec2, vec3

deg45 = math.sqrt(2)/2

flow_list = [[[deg45,deg45],[0,1.0],[0,1.0],[-deg45,deg45]],
             [[1.0,0],[deg45,deg45],[-deg45,deg45],[-1.0,0]],
             [[1.0,0],[deg45,-deg45],[-deg45,-deg45],[-1.0,0]],
             [[deg45,-deg45],[0,-1.0],[0,-1.0],[-deg45,-deg45]]]

@ti.data_oriented
class Shader:
    def __init__(self, app):
        self.app = app
        
        self.screen_field = ti.Vector.field(3, ti.f32, (self.app.vector_width, self.app.vector_height))
        self.agent_field = ti.vector.field(2, ti.int16, (self.app.width, self.app.height))
        
    @ti.kernel
    def calc(self):
        for frag_coord in ti.grouped(self.screen_field):
            uv = frag_coord/self.app.vector_field.xy
            
            c = self.SmoothNoise64(uv)
            
            col = vec3(c)
            
            self.screen_field[frag_coord.x, self.app.vector_field.y - frag_coord.y - 1] = col * 255
    def Noise21(self, uv: vec2):
        return ti.math.fract(ti.math.sin(uv.x * 100. + uv.y*6124)*5674)
    
    def SmoothNoise(self, uv: vec2):
        lv = ti.math.fract(uv)
        id = ti.math.floor(uv)
            
        lv = lv*lv*(3.0 -2.0*lv)
        
        bl = self.Noise21(id)
        br = self.Noise21(id+vec2(1,0))
        b = ti.math.mix(bl,br,lv.x)
            
        tl = self.Noise21(id+vec2(0,1))
        tr = self.Noise21(id+vec2(1,1))
        t = ti.math.mix(tl,tr,lv.x)
            
        return ti.math.mix(b,t, lv.y)

    def SmoothNoise64(self, uv: vec2):
        c = self.SmoothNoise(uv*4)
        c = c + self.SmoothNoise(uv*8)*(1/2)
        c = c + self.SmoothNoise(uv*16)*(1/4)
        c = c + self.SmoothNoise(uv*32)*(1/8)
        c = c + self.SmoothNoise(uv*64)*(1/16)
        c = c + self.SmoothNoise(uv*128)*(1/32)
        c = c + self.SmoothNoise(uv*256)*(1/64)
        c = c / 2
        return c 
    
    def update(self):
        self.calc()
        #self.app.screen_array = self.screen_field.to_numpy()

    def draw(self):
        pg.surfarray.blit_array(self.app.display, self.app.screen_array)

    def run(self):
        self.update()
        #self.draw()

class App:
    def __init__(self):
        pg.init()
        ti.init(arch=ti.vulkan)
        
        self.resolution = self.width, self.height = vec2(800, 800)
        self.vector_field = self.vector_width, self.vector_height = vec2(800, 800)
        
        self.screen_array = np.full((self.vector_width,self.vector_height),0, np.float32)
        self.agent_array = np.full((self.vector_width,self.vector_height,2),[0,0], np.uint16)
        
        self.screen = pg.display.set_mode(self.resolution)
        self.display = pg.Surface(self.vector_field)
        
        self.clock = pg.time.Clock()
        self.shader = Shader(self)
        
    def run(self):
        start  = [200,50]
        while True:
            self.shader.run()
            #start[0] += flow_list[int(start[0]/100)][int(start[1]/100)][0]*1
            #start[1] += flow_list[int(start[0]/100)][int(start[1]/100)][1]*1*-1

            #pg.draw.circle(self.screen, (255,0,0), (start[0],start[1]), 1)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

            #surf = pg.transform.scale(self.display, self.resolution)
            #self.screen.blit(surf, (0,0))
            
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')
            self.clock.tick()
            pg.display.flip()

if __name__ == '__main__':
    app = App()
    app.run()