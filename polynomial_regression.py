import numpy.random as rnd
x=6*rnd.rand(100,1)-3
y=0.5*x**2+x+2+rnd.randn(100,1)


from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2, include_bias=False)
x_poly=poly_features.fit_transform(x)
x[0]

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
lin_reg.intercept_,lin_reg.coef_

x=6*rnd.rand(100,1)-3
y=0.5*x**2+x+2+rnd.randn(100,1)

import numpy.random as rnd
x=6*rnd.rand(100,1)-3
y=0.5*x**2+x+2+rnd.randn(100,1)


from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2, include_bias=False)
x_poly=poly_features.fit_transform(x)
x[0]

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()


lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)
lin_reg.intercept_,lin_reg.coef_

u=0.47815515*x**2+1.0270022*x+2.17485908


plt.figure(figsize=(10,5))
plt.scatter(x,y)
plt.scatter(x,u)