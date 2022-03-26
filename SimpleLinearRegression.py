import pandas as pd
import matplotlib.pyplot as plt
# for graph

data = pd.read_csv('hoursVSsalary.csv')
# reading data from csv file

def algorithm(b0, b1, data): #loss function
    total_error = 0
    for i in range(len(data)):
        x = data.iloc[i].hours
        y = data.iloc[i].salary
        total_error += (y - (b0 * x + b1)) ** 2  # basically like (err = y - yc)^2
    total_error / float(len(data)) # average total error
    

def grad_descent(b0_curr, b1_curr, data, alpha): # merges algorithm above and updates betas
    b0_grad = 0
    b1_grad = 0
    
    n = len(data)
    
    for i in range(n):
        x = data.iloc[i].hours
        y = data.iloc[i].salary
        
        b0_grad += -(2/n) * x * (y - (b0_curr * x + b1_curr)) # formula for gradual descent
        b1_grad += -(2/n) * (y - (b0_curr * x + b1_curr))
    
    b0 = b0_curr - b0_grad * alpha
    b1 = b1_curr - b1_grad * alpha
    return b0, b1

b0 = 0 # beta m
b1 = 0 # beta b (slope)
alpha = 0.12 # learningRate
epochs = 2500 # iterations

for i in range(epochs):
    if i%50 == 0:
        print(f"Epoch {i} :",b0,b1)
    b0, b1 = grad_descent(b0, b1, data, alpha)
   
    
print("Final beta values: (",b0,b1,")")

# plotting values
plt.scatter(data.hours, data.salary, color="black")
plt.plot([x*0.01 for x in range(0,120)], [(b0 * x + b1)*0.01 for x in range(0,120)], color="red")
plt.xlabel("Hours Worked")
plt.ylabel("Salary")
plt.show() 

