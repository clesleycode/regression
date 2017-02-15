Intro to Regression Analysis 
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958).

## Table of Contents

- [0.0 Setup](#00-setup)
	+ [0.1 R and R Studio](#01-r-and-r-studio)
	+ [0.2 Packages](#02-packages)
- [1.0 Review](#10-review)
- [2.0 Data Preparation](#30-data-preparation)
	+ [2.1 dplyr](#31-dplyr)
	+ [2.2 Geopandas](#32-geopandas)
- [3.0 Exploratory Analysis](#30-exploratory-analysis)
- [4.0 Data Visualization](#50-data-visualization)
- [5.0 Machine Learning & Prediction](#50-machine-learning--prediction)
	+ [5.1 Random Forests](#51-random-forests)
	+ [5.2 Natural Language Processing](#52-natural-language-processing)
		* [5.2.1 ANLP](#521-anlp)
	+ [5.3 K Means Clustering](#53-k-means-clustering)
- [6.0 Final Exercise]($60-final-exercise)
- [7.0 Final Words](#60-final-words)
	+ [7.1 Resources](#61-resources)
	+ [7.2 More!](#72-more)

## 0.0 Setup

This guide was written in R 3.2.3.


### 0.1 R and R Studio

Download [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).


### 0.2 Packages

Next, to install the R packages, cd into your workspace, and enter the following, very simple, command into your bash: 

```
R
```

This will prompt a session in R! From here, you can install any needed packages. For the sake of this tutorial, enter the following into your terminal R session:

```
install.packages("")
```

## 1.0 Review


### Random Variables

Values of this variable are different every time it's observed. To denote these, capital letters are used. Lower cases are reserved for actual observed values. 

So when we say `P(H > h)`, where the random variables refer to heights, what we're asking for is the probability that the height is larger than some observed height <i>h</i>.

### Probability Distribution

The probability distribution describes the distribution of a random variable and is defined by its density function, `f(h)`. Note that the area under the distribution gives us the probability.

### Correlation Coefficient

The correlation coefficient, <b>r</b> indicates the nature and strength of the relationship betwee x and y. Values of r range from -1 to +1. A correlation coefficient of 0 indicates there is no relationship.

## 2.0 Linear Regression

### 2.1 Basic Equation
The variable that we want to predict, `x`, is called the independent variable. We can collect values of y for known values of x in order to derive the co-efficient and y-intercept of the model using certain assumptions. The equation looks like below:

``` 
y = a + bx + e
```
Here, `a` is the y-intercept, `b` is the slope of the line, and `e` is the error term. Usually we don't know the error term, so we reduce this equation to:

```
y = a + bx
```

### 2.2 Error Term

The difference between the observed value of the dependent variable and the predicted value is called the error term, or residual. Each data point has its own residual.

When a residual plot shows a random pattern, it indicated a good fit for a linear model.

### 2.3 Assumptions

There are four assumptions that allow for the use of linear regression models. If any of these assumptions is violated, then the forecasts, confidence intervals, and insights yielded by a regression model may be inefficient, biased, or misleading. 

#### 2.3.1 Linearity

The first assumption is the linearity and additivity between dependent and independent variables. Because of this assumption, the expected value of dependent variable is a straight-line function of each independent variable, holding the others fixed. Lastly, the slope of this doesn't depend on the other variables. 

#### 2.3.2 Statistical Independence

The statistical independence of the errors means there is no correlation between consecutive errors.

#### 2.3.3 Homoscedasticity

This refers to the idea that there is a constant variance of errors. This is true against time, predictions, and any independent variable. 

#### 2.3.4 Error Distribution

This says that the distribution of errors is normal.


### 2.4 Correlation Coefficient 

The standardized correlation coefficient is the same as Pearson's correlation coefficient. While correlation typically refers to Pearson’s correlation coefficient, there are other types of correlation, such as Spearman’s.

### 2.5 Disadvantages

Firstly, if the data doesn't follow the normal distribution, the validity of the regression model suffers. 

Secondly, there can be collinearity problems, meaning if two or more independent variables are strongly correlated, they will eat into each other's predictive power. 

Thirdly, if a large number of variables are included, the model may become unreliable. Regressions doesn't automatically take care of collinearity.

Lastly, regression doesn’t work with categorical variables with multiple values. These variables need to be converted to other variables before using them in regression models.

## Logistic Regression

Logistic regression is a generalized linear model most commonly used for classifying binary data.  Its output is a continuous range of values between 0 and 1 (commonly representing the probability of some event occurring), and its input can be a multitude of real-valued and discrete predictors.



## 5.0 Final Words


### 5.1 Resources

[]() <br>
[]()
