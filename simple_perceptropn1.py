# 퍼셉트론 
# AND(논리곱) 게이트
w1 = 0.5
w2 = 0.5
b = 0.0

def perceptron(x1, x2):
    y = w1 * x1 + w2 * x2 + b   
    if y > 0.5:
        return 1
    else:
        return 0
    
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print("입력: ", x1, x2, " 출력", perceptron(x1, x2))

