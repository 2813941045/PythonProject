import turtle
import time


# 清屏函数
def clear_all():
    turtle.penup()
    turtle.goto(0, 0)
    turtle.color('white')
    turtle.pensize(800)
    turtle.pendown()
    turtle.setheading(0)
    turtle.fd(300)
    turtle.bk(600)


# 重定位海龟的位置
def go_to(x, y, state):
    turtle.pendown() if state else turtle.penup()
    turtle.goto(x, y)


# 画爱心
def draw_heart(size):
    turtle.color('red', 'pink')
    turtle.pensize(2)
    turtle.pendown()
    turtle.setheading(150)
    turtle.begin_fill()
    turtle.fd(size)
    turtle.circle(size * -3.745, 45)
    turtle.circle(size * -1.431, 165)
    turtle.left(120)
    turtle.circle(size * -1.431, 165)
    turtle.circle(size * -3.745, 45)
    turtle.fd(size)
    turtle.end_fill()


# 第一个画面，显示文字
def paintingOne():
    turtle.penup()
    turtle.goto(-300, 0)
    turtle.color('pink')
    turtle.write('时光让我们相遇，我的情人，七夕快乐！！！', font=('楷体', 24, 'normal'))
    time.sleep(3)


# 画出人物
def draw_people(x, y):
    turtle.penup()
    turtle.goto(x, y)
    turtle.pendown()

    turtle.pensize(2)
    turtle.color('pink')

    turtle.setheading(0)
    turtle.circle(60, 360)

    turtle.penup()
    turtle.setheading(90)
    turtle.fd(75)

    turtle.setheading(180)
    turtle.fd(20)

    turtle.pensize(4)
    turtle.pendown()

    turtle.circle(2, 360)
    turtle.setheading(0)

    turtle.penup()
    turtle.fd(40)
    turtle.pensize(4)
    turtle.pendown()
    turtle.circle(-2, 360)

    turtle.penup()
    turtle.goto(x, y)
    turtle.setheading(-90)
    turtle.pendown()

    turtle.fd(20)
    turtle.setheading(0)
    turtle.fd(35)
    turtle.setheading(60)
    turtle.fd(10)

    turtle.penup()
    turtle.goto(x, y)
    turtle.setheading(-90)
    turtle.pendown()

    turtle.fd(40)
    turtle.setheading(0)
    turtle.fd(35)
    turtle.setheading(-60)
    turtle.fd(10)

    turtle.penup()
    turtle.goto(x, y)
    turtle.setheading(-90)
    turtle.pendown()
    turtle.fd(60)
    turtle.setheading(-135)

    turtle.fd(60)
    turtle.bk(60)
    turtle.setheading(-45)

    turtle.fd(30)
    turtle.setheading(-135)

    turtle.fd(35)
    turtle.penup()


# 第二个画面，显示发射爱心的小人
def paintingTwo():
    turtle.speed(10)

    draw_people(-250, 20)

    turtle.penup()
    turtle.goto(-150, -30)
    draw_heart(14)

    turtle.penup()
    turtle.goto(-20, -60)
    draw_heart(25)

    turtle.penup()
    turtle.goto(250, -100)

    draw_heart(45)

    turtle.hideturtle()
    time.sleep(1)


def Main():
    turtle.setup(900, 500)
    paintingOne()
    clear_all()

    paintingTwo()
    clear_all()

    turtle.done()


if __name__ == '__main__':
    Main()

