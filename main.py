import pygame as pg
import sys
import numpy as np
import taichi as ti
from taichi_glsl import vec2, vec3
import random

@ti.data_oriented
class Shader:
    def __init__(self, app):
        self.app = app
        
        self.screen_field = ti.field(ti.f32, (self.app.vector_width, self.app.vector_height))
        self.agent_field = ti.Vector.field(3, ti.int16, (self.app.width, self.app.height))
        self.agents_field = ti.Vector.field(2, ti.int16, self.app.agent_num)
        i = 0
        for i in range(0, self.app.agent_num):
            self.agents_field[i] = vec2(random.randint(1,799), random.randint(1,799))
        
    @ti.kernel
    def calc(self, time: ti.float32):
        for frag_coord in ti.grouped(self.screen_field):
            uv = frag_coord/self.app.vector_field.xy
            
            uv.x = uv.x + time*0.1
            uv.y = uv.y + time*0.5
            
            c = self.SmoothNoise64(uv)
            
            col = c
            
            self.screen_field[frag_coord.x, self.app.vector_field.y - frag_coord.y - 1] = col * 360
        for i in range(0, self.app.agent_num):
            self.agent_field[self.agents_field[i].x, self.agents_field[i].y] = vec3(255)
            
            self.agents_field[i].x = self.agents_field[i].x+1*ti.math.sin(self.screen_field[self.agents_field[i].x,self.agents_field[i].y])
            self.agents_field[i].y = self.agents_field[i].y+1*ti.math.cos(self.screen_field[self.agents_field[i].x,self.agents_field[i].y])
            if self.agents_field[i].x <= 0:
                self.agents_field[i].x = 799
            if self.agents_field[i].x >= 800:
                self.agents_field[i].x = 1
            if self.agents_field[i].y <= 0:
                self.agents_field[i].y = 799
            if self.agents_field[i].y >= 800:
                self.agents_field[i].y = 1
        for frag_coords in ti.grouped(self.agent_field):
            if self.agent_field[frag_coords.x, self.app.vector_field.y - frag_coords.y - 1][1] > 0:
            
                self.agent_field[frag_coords.x, self.app.vector_field.y - frag_coords.y - 1] -= vec3(1)
        
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
    
    def hash(self, seed, max):
        a = 1103515245
        c = 12345
        m = 2 ** 32
        x = seed
        x = (a * x + c) % m
        return int(x % max)
    
    def update(self):
        time = pg.time.get_ticks()* 1e-3
        self.calc(time)
        self.app.screen_array = self.agent_field.to_numpy()

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.app.screen_array)

    def run(self):
        self.update()
        self.draw()

class App:
    def __init__(self):
        pg.init()
        ti.init(arch=ti.vulkan)
        
        self.resolution = self.width, self.height = vec2(800, 800)
        self.vector_field = self.vector_width, self.vector_height = vec2(800, 800)
        self.agent_num = 1000
        
        self.screen_array = np.full((self.vector_width,self.vector_height),0, np.float32)
        self.agent_array = np.full((self.vector_width,self.vector_height,2),[0,0], np.uint16)
        self.screen = pg.display.set_mode(self.resolution)
        
        self.clock = pg.time.Clock()
        self.shader = Shader(self)
        
    def run(self):
        while True:
            self.shader.run()
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
            
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')
            self.clock.tick()
            pg.display.flip()

if __name__ == '__main__':
    app = App()
    app.run()