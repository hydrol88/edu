# 2 layers와 2개의 병렬로 연결한 펜센트론
# (XOR 문제 해결 )
def perceptron1(x1, x2):
    w1 = 5
    w2 = 5
    b = -8
    y1 = w1 * x1 + w2 * x2 + b   
    if y1 > 0.5:
        y1 = 1
        return y1
    else:
        y1 = 0
        return y1
    
def perceptron2(x1, x2):
    w1 = -7
    w2 = -7
    b = 3
    y2 = w1 * x1 + w2 * x2 + b   
    if y2 > 0.5:
        y2 = 1
        return y2
    else:
        y2 = 0
        return y2
    
def perceptron3( y1, y2):
    w1 = -11
    w2 = -11
    b = 6
    y3 = w1 * x1 + w2 * x2 + b   
    if y3 > 0.5:
        y3 = 1
        return y3
    else:
        y3 = 0
        return y3
    
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    y1 = perceptron1(x1,x2)
    y2 = perceptron2(x1,x2)
    y3 = perceptron2(y1,y2)
    print("입력: ", x1, x2, " 출력", y3)
