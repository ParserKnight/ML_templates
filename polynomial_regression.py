import numpy.random as rnd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def run_example():
    """Function to exhibit an example of polynomial regression"""

    #creation of random x,y dataset
    x = 6*rnd.rand(100, 1)-3
    y = 0.5*x**2+x+2+rnd.randn(100, 1)

    # Generation a new feature matrix consisting of all polynomial combinations of\
    # the features with degree less than or equal to the specified degree
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly_features.fit_transform(x)

    # Curve equation resolver 
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    u = lin_reg.coef_[0][1]*x**2+lin_reg.coef_[0][0]*x+lin_reg.intercept_[0]

    # Plot of curves
    plt.figure(figsize = (10, 5))
    plt.scatter(x,y)
    plt.scatter(x,u)
    plt.savefig('resultado.png')

if __name__=="__main__":
    run_example()