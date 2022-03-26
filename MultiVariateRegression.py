import pandas as pd
import matplotlib.pyplot as plt
# for graph

data = pd.read_csv('andGate.csv')
# reading data from csv file

    

def grad_descent(b0_curr, b1_curr, b2_curr, data, alpha): # merges algorithm above and updates betas
    b0_grad = 0 #slope
    b1_grad = 0 #x1
    b2_grad = 0 #x2
    n = len(data)
    
    for i in range(n):
        x1 = data.iloc[i].bit0
        x2 = data.iloc[i].bit1
        y = data.iloc[i].result
        
        b0_grad += -(2/n) * (y - (b0_curr + b1_curr * x1 + b2_curr * x2)) # formula for gradual descent
        b1_grad += -(2/n) * x1 * (y - (b0_curr + b1_curr * x1 + b2_curr * x2))
        b2_grad += -(2/n) * x2 * (y - (b0_curr + b1_curr * x1 + b2_curr * x2))
        
    b0 = b0_curr - b0_grad * alpha
    b1 = b1_curr - b1_grad * alpha
    b2 = b2_curr - b2_grad * alpha
    return b0, b1, b2

b0 = 0 
b1 = 0 
b2 = 0
alpha = 0.12 # learningRate
epochs = 2500 # iterations

for i in range(epochs):
    if i%50 == 0:
        print(f"Epoch {i} :",b0,b1)
    b0, b1, b2 = grad_descent(b0, b1, b2, data, alpha)
   
    
print("Final beta values: (",b0,b1,b2, ")")

# plotting values
plt.scatter(data.bit0, data.bit1, data.result, color="black")
a = [1,1,0,0]
b = [1,0,1,0]
plt.plot(list(x for x in range(0,4)), [b0 + b1*aa + b2*bb for aa,bb in zip(a,b) ] , color="red")

plt.show() 

print("Test Values: ")
print("Bit 1: ", end="")
testbit0 = float(input())
print("Bit 2: ", end="")
testbit1 = float(input())
output = (b0 + b1 * testbit0 + b2 * testbit1)
print("Output: ",  output)
