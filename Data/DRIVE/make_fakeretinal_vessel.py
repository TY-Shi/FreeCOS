#from turtle import *
import turtle
import numpy as np
import random
import math
from PIL import Image
#tempNumpyArray=np.load("DRIVE_instensity.npy")
#score = tempNumpyArray.tolist()

def draw_background(a_turtle):
    """ Draw a background rectangle. """
    ts = a_turtle.getscreen()
    canvas = ts.getcanvas()
    height = ts.getcanvas()._canvas.winfo_height()
    width = ts.getcanvas()._canvas.winfo_width()

    turtleheading = a_turtle.heading()
    turtlespeed = a_turtle.speed()
    penposn = a_turtle.position()
    penstate = a_turtle.pen()

    a_turtle.penup()
    a_turtle.speed(0)  # fastest
    a_turtle.goto(-width/2-2, -height/2+3)
    print("turtle.Screen().bgcolor()",turtle.Screen().bgcolor())
    a_turtle.fillcolor(255,255,255)
    a_turtle.begin_fill()
    a_turtle.setheading(0)
    a_turtle.forward(width)
    a_turtle.setheading(90)
    a_turtle.forward(height)
    a_turtle.setheading(180)
    a_turtle.forward(width)
    a_turtle.setheading(270)
    a_turtle.forward(height)
    a_turtle.end_fill()

    a_turtle.penup()
    a_turtle.setposition(*penposn)
    a_turtle.pen(penstate)
    a_turtle.setheading(turtleheading)
    a_turtle.speed(turtlespeed)

def makecolor(turtle,new_length,width,x,y,theta):
    #Tile one pixel or two pixel
    turtle_screen.colormode(255)
    # system.draw(turtle)
    #Need more set of different colorset#Get random colorset
    random_R = np.random.randint(0,255)
    random_G = np.random.randint(0, 255)
    random_B = np.random.randint(0, 255)
    R_insity = random_R
    G_insity = random_G
    B_insity = random_B

    Ar1 = np.random.uniform(-3,3)
    Ag1 = np.random.uniform(-3,3)
    Ab1 = np.random.uniform(-3,3)
    turtle.pu()
    turtle.goto((x, y))
    turtle.setheading(theta)
    width = width
    if width<2:
        width = 2
    turtle.pensize(width)
    #init_width too thin single guassian<=2
    new_instenisty_r = int(np.clip(R_insity + Ar1,0,255))
    new_instenisty_g = int(np.clip(G_insity + Ag1,0,255))
    new_instenisty_b = int(np.clip(B_insity + Ab1,0,255))
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    #init_width Too thick with bin_guassian>2
    turtle.pd()
    turtle.forward(new_length)
    turtle.pu()
    turtle.goto((x, y))
    turtle.setheading(theta)
    turtle.pd()
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    turtle.forward(new_length)

class LSystem_vessel():
    def __init__(self, axiom, rules, rules_2, rules_3, theta=0, width=5, dtheta_1=40, dtheta_2=30, start=(-350,0), length=80,iteration=3,width_1=0.79, width_2 = 0.5):
        self.sentence = axiom
        self.rules = rules
        self.relus_2 = rules_2
        self.rules_3 = rules_3
        self.iteration = iteration
        self.width = width
        self.theta = theta
        self.dtheta_1 = dtheta_1
        self.dtheta_2 = dtheta_2
        self.length = length
        self.positions = []
        self.start = start
        self.lamda_1 =  width_1
        self.lamda_2 = width_2

        self.x, self.y = start

    def __str__(self):
        return self.sentence

    def generate(self):
        self.x, self.y = self.start
        for iter in range(self.iteration):
            newStr = ""
            for char in self.sentence:
                mapped = char
                try:
                    p = random.random()
                    if p>0.4:
                        mapped = self.rules[char]
                    elif 0.8>p>=0.4:
                        mapped = self.rules_3[char]
                    else:
                        mapped = self.relus_2[char]
                except:
                    pass
                newStr += mapped
            self.sentence = newStr

    def draw(self, turtle):
        turtle.pu()
        turtle.hideturtle()
        turtle.speed(0)
        turtle.goto((self.x, self.y))
        turtle.setheading(self.theta)
        flag = False
        for char in self.sentence:
            turtle.pd()
            if char == 'F' or char == 'G':
                turtle.pu()
                turtle.setheading(self.theta)
                if flag == True:
                    new_length = self.length*self.lamda_1
                else:
                    new_length = self.length*self.lamda_2
                turtle.pensize(self.width)
                x, y = turtle.position()
                theta = self.theta
                width = self.width
                makecolor(turtle, new_length, width, x, y, theta)
                self.x, self.y = turtle.position()
            elif char == '+':
                dtheta = np.random.randint(low=10, high=40)
                self.theta += dtheta
                self.width = self.width * self.lamda_1
                turtle.right(self.theta)
            elif char == '-':
                self.width = self.width * self.lamda_2
                dtheta = np.random.randint(low=10, high=40)
                self.theta -= dtheta
                turtle.left(self.theta)
            elif char == '[':
                self.positions.append({'x': self.x, 'y': self.y, 'theta': self.theta, 'width': self.width, 'length': self.length,"new_width_0": self.width * 0.79})
                flag = True
            elif char == ']':
                turtle.pu()
                position = self.positions.pop()
                self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                flag = False
                turtle.goto((self.x, self.y))
                turtle.setheading(self.theta)


rules = {"F":"F-F[+F-F][-F+F]"}
rules_2 = {"F":"F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"}
rules_3 = {"F":"F[+F]+F[+F]+F[+F]"}
rules_4 = {"F":"F-F-F[+F-F][-F-F]F+F"}

path = "[+F-F][-F]"
print(path)

Num_image = 600

Start_theta = (-80,20)
Start_theta_2 = (170,210)
Start_position_x = (-350, -150)
Start_position_y = (-100, -100)
Start_position_x2 = (150, 350)
Ratio_LW = (0.7,1)

Dtheta = (20,120)
Width = (2,6)
Length_range = (90, 150)

for i  in range(Num_image):
    p2 = random.random()
    if p2>0.5:
        path = "[+F-F][-F]"
    else:
        path = "F"
    p_vessel = random.random()
    if p_vessel>0.5:
        r1= rules
        r2= rules_2
        r3 = rules_3
    else:
        r1 = rules_4
        r2 = rules_3
        r3 = rules
    print("image_num", i)
    dtheta_1 = np.random.uniform(Dtheta[0],Dtheta[1])
    dtheta_2 = np.random.uniform(Dtheta[0],Dtheta[1])
    init_width = np.random.randint(Width[0],Width[1]+1)
    init_length = np.random.randint(Length_range[0],Length_range[1]+1)
    iteration = np.random.randint(1,3)
    #iteration = 1

    Ratio_lw_1 = np.random.uniform(Ratio_LW[0],Ratio_LW[1])
    Ratio_lw_2 = np.random.uniform(Ratio_LW[0], Ratio_LW[1])

    p = random.random()
    if p>0.5:
        init_theta = np.random.uniform(Start_theta[0], Start_theta[1])
        x_position = np.random.randint(Start_position_x[0],Start_position_x[1]+1)
    else:
        init_theta = np.random.uniform(Start_theta_2[0], Start_theta_2[1])
        x_position = np.random.randint(Start_position_x2[0],Start_position_x2[1]+1)
    y_position = np.random.randint(Start_position_y[0],Start_position_y[1]+1)

    system = LSystem_vessel(path, r1, r2, r3, theta=init_theta, width=init_width, dtheta_1=dtheta_1, dtheta_2=dtheta_2, start=(x_position, y_position),length=init_length, iteration=iteration, width_1=Ratio_lw_1, width_2=Ratio_lw_2)
    system.generate()
    turtle.speed(0)
    turtle.delay(0)
    turtle.tracer(False)
    turtle_screen = turtle.Screen()  # create graphics window
    turtle_screen.colormode(255)
    #ratip=1.36
    turtle.setup(800,800)
    turtle_screen.screensize(800, 800)
    turtle.bgcolor(0,0,0)
    draw_background(turtle)
    system.draw(turtle)
    file_name = "./train/fake_onlythin_vessel/"+str(i)+'.png'
    tsimg = turtle.getscreen()
    tsimg.getcanvas().postscript(file="work_vessel.eps")
    im = Image.open("work_vessel.eps")
    out = im.resize((512,512))
    #out = im
    im_array = np.array(out)
    print("im_array",im_array.shape)
    out.save(file_name)
    turtle.reset()