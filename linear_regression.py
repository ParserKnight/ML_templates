from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import numpy as np
import numpy.random as rnd
import matplotlib
import matplotlib.pyplot as plt

def run_example():
    """Function to exhibit an example of linear regression"""

    mnemo_linreg=LinearRegression()

    x=3*rnd.rand(100,1)
    y=4+(3*x)+rnd.randn(100,1)

    plt.figure(figsize=(20,10))
    plt.scatter(x,y)

    mnemo_linreg.fit(x,y)
    #mnemo_linreg.intercept_, mnemo_linreg.coef_

    regresion=4.03194963+2.96077091*x

    plt.figure(figsize=(20,10))
    plt.scatter(x,y)
    plt.plot(x,regresion, 'purple')

    sgd_reg=SGDRegressor(n_iter=50, penalty=None, eta0=0.1)

    sgd_reg.fit(x,y)
    sgd_reg.intercept_,sgd_reg.coef_

    regresion_sgd=4.0402834+2.97237333*x

    plt.figure(figsize=(20,10))
    plt.scatter(x,y)
    plt.plot(x,regresion_sgd, 'r-')

    sgd_reg.predict(36)
    sgd_reg.predict(1)

if __name__=="__main__":
    run_example()