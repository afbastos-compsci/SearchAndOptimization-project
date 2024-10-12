import numpy as np
import matplotlib.pyplot as plt
import statistics

f1 = "(x1**2)+(x2**2)"
f2 = "np.exp(-1*(x1**2 + x2**2))+2*np.exp(-((x1-1.7)**2+(x2-1.7)**2))"
f3 = "-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))) + 20 + np.exp(1)"
f4 = "(x1**2 - 10 * np.cos(2*np.pi*x1) + 10) + (x2**2 - 10 * np.cos(2*np.pi*x2)+10)"
f5 = "((x1 * np.cos(x1)) / 20) + 2 * np.exp(-(x1**2) - ((x2-1)**2)) + 0.01 * x1 * x2"
f6 = "x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1"
f7 = "(- np.sin(x1)) * np.sin(x1**2 / np.pi)**(2*10) - np.sin(x2) * np.sin(x2**2 / np.pi)**(2*10)"
f8 = "-(x2 + 47) * np.sin(np.sqrt(abs(x1/2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))"

def hill_climbing(x,y,func,objetivo):

    def perturb(x:tuple[float, float],e:float,x_axis,y_axis):
    
        x[0] = np.clip(np.random.uniform(low=x[0]-e,high=x[0]+e),x_axis[0],x_axis[1])
        x[1] = np.clip(np.random.uniform(low=x[1]-e,high=x[1]+e),y_axis[0],y_axis[1])
        return x

    def f(func,x1:float,x2:float):
        return eval(func,{'np':np},{'x1':x1,'x2':x2})

    x_axis = np.linspace(x[0],x[1],1000)
    y_axis = np.linspace(y[0],y[1],1000)
    X,Y = np.meshgrid(x_axis,y_axis)

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    #ax.plot_surface(X,Y,f(func,X,Y),cmap='jet', alpha=.6,rstride=30,cstride=30)

    x_opt = [x[0],y[0]]
    f_opt = f(func,x_opt[0],x_opt[1])

    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')



    max_it = 10000
    max_viz = 5
    e = .1

    i = 0
    melhoria = True

    while i < max_it and melhoria:
        melhoria = False
        for j in range(max_viz):
            x_cand = perturb(np.copy(x_opt),e,x,y)
            f_cand = f(func,x_cand[0],x_cand[1])

            if(objetivo == "max"):
                if f_cand > f_opt:
                    x_opt = x_cand
                    f_opt = f_cand
                    melhoria = True
                    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')
                    break

            elif(objetivo == "min"):
                if f_cand < f_opt:
                    x_opt = x_cand
                    f_opt = f_cand
                    melhoria = True
                    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')
                    break
        i+=1
    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='g')
                
    #plt.show()

    return f_opt

def lrs(x, y, func, objetivo):

    x_l = np.array([x[0], y[0]]) 
    x_u = np.array([x[1], y[1]])  

   
    
    def f(func,x1,x2):
        return eval(func,{'np':np},{'x1':x1,'x2':x2})
    

    x_opt = np.random.uniform(x_l, x_u)
    f_opt = f(func,x_opt[0],x_opt[1])
    
    # Configuração do gráfico
    x_axis = np.linspace(x[0], x[1], 1000)
    y_axis = np.linspace(y[0], y[1], 1000)
    X,Y = np.meshgrid(x_axis,y_axis)
    
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    #ax.plot_surface(X,Y,f(func,X,Y),cmap='jet', alpha=.6,rstride=30,cstride=30)
    
    max_it = 10000
    max_viz = 5

    sigma = 1

    i = 0    
    melhoria = True
    
    while i < max_it and melhoria:
        melhoria = False
        for j in range(max_viz):
            n = np.random.normal(0, sigma, size=x_opt.shape)
            x_cand = x_opt + n
            x_cand = np.clip(x_cand, x_l, x_u)
            f_cand = f(func,x_cand[0],x_cand[1])

            if(objetivo == "max"):
                if f_cand > f_opt:
                    x_opt = x_cand
                    f_opt = f_cand
                    melhoria = True
                    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')
                    #plt.pause(.1)
                    break

            elif(objetivo == "min"):
                if f_cand < f_opt:
                    x_opt = x_cand
                    f_opt = f_cand
                    melhoria = True
                    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')
                    #plt.pause(.1)
                    break
        i+=1
    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='g')
                
    #plt.show()
    
    return f_opt

def grs(x, y, func, objetivo):

    x_l = np.array([x[0], y[0]]) 
    x_u = np.array([x[1], y[1]])  

   
    
    def f(func,x1,x2):
        return eval(func,{'np':np},{'x1':x1,'x2':x2})
    

    x_opt = np.random.uniform(x_l, x_u)
    f_opt = f(func,x_opt[0],x_opt[1])
    
    # Configuração do gráfico
    x_axis = np.linspace(x[0], x[1], 1000)
    y_axis = np.linspace(y[0], y[1], 1000)
    X,Y = np.meshgrid(x_axis,y_axis)
    
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')

    #ax.plot_surface(X,Y,f(func,X,Y),cmap='jet', alpha=.6,rstride=30,cstride=30)
    
    max_it = 10000
    #max_viz = 5

    sigma = 1

    i = 0    
    melhoria = True
    
    while i < max_it and melhoria:
        melhoria = False

        n = np.random.normal(0, sigma, size=x_opt.shape)
        x_cand = x_opt + n
        x_cand = np.clip(x_cand, x_l, x_u)
        f_cand = f(func,x_cand[0],x_cand[1])

        if(objetivo == "max"):
            if f_cand > f_opt:
                x_opt = x_cand
                f_opt = f_cand
                melhoria = True
                #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')
                #plt.pause(.1)
                break

        elif(objetivo == "min"):
            if f_cand < f_opt:
                x_opt = x_cand
                f_opt = f_cand
                melhoria = True
                #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='r')
                #plt.pause(.1)
                break
        i+=1
    #ax.scatter(x_opt[0],x_opt[1],f_opt,marker='x',color='g')
                
    #plt.show()
    
    return f_opt

#hill_climbing([-100,100],[-100,100],f1,'min')
#hill_climbing([-2,4],[-2,5],f2,'max')
#hill_climbing([-8,8],[-8,8],f3,'min')
#hill_climbing([-5.12,5.12],[-5.12,5.12],f4,'min')


result =[0]*100

for i in range(len(result)):
    result[i] = lrs([-100,100],[-100,100],f1,'min').item()

print(result)
print(f"Moda da função: {statistics.mode(result)}")