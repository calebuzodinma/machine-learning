# machine-learning
Using machine learning to predict energy consumption.

This report discusses the challenges of predicting energy consumption in residential buildings in
order to promote greater energy efficiency and conservation. For most parts, sustainability has
become the way to go in the energy sector. There have been many sustainable initiatives such as
electrifying public transportation and building electric vehicle infrastructure. For this report,
predicting energy consumption in a bid to conserve energy and bring greater energy efficiency
will be challenges that will be discussed.
This study used a dataset from the government of Canada, which contained energy consumption
data of households in various provinces of Canada. The dataset for this project used electricity as
its main energy source and residential buildings as its case study for infrastructure. Other energy
sources, such as oil, gas, and solar, were not used as adding them would be too broad to quantify
and would need a considerable amount of time to analyze. This was done to narrow down the
datasets and to avoid any sort of complexity. Linear regression, Support vector regression, and
polynomial regression were the machine learning methods used to predict the energy
consumption forecast.
The methodology of the project followed a sequence of steps: data collection, data
preprocessing, algorithm selection, and model evaluation. Model validation techniques, such as
train-test split and performance evaluation, were employed to ensure that the model can perform
well with new data. The limitations of the study include the absence of weather data and the use
of deep learning to reduce bias. Overall, at least two models were found to be predictive and
helped conserve energy.

2 – Introduction

Nowadays, the climate of the environment has become a significant concern to a large
population. Finding ways to be sustainable is one of the biggest issues that plague our society
today. Energy conservation is a step in the right direction towards addressing the climate crisis.
The quest for greater energy efficiency has united people around the globe over the past year,
driven by the concurrent challenges of the energy and climate crises [1]. There is now a sense of
eagerness and urgency to save electricity.
The International Energy Agency (IEA) notes that since 2020, around $1 trillion has been
mobilized to boost energy efficiency, covering initiatives as varied as making homes and
commercial buildings more energy efficient, electrifying public transportation and building
electric vehicle infrastructure [1]. One of the modern initiatives that can help save energy is
using machine learning algorithms to forecast the energy consumption in residential buildings.
Using machine learning algorithms is a modern approach that helps show the difference between
the projected energy use and the actual energy consumption.
This report provides an analysis on the energy consumption in Residential areas. Most homes
tend to waste energy, this can lead to pollution if not addressed. Therefore, homeowners would
need to resolve the waste of energy. An energy consumption forecast will be able to resolve this
issue. Homeowners can use the differences between the projected consumption and actual energy
to identify areas where they can conserve energy in the building.
This report will show how three machine learning algorithms will predict energy consumption in
residential buildings. Each algorithm will be explained to give an understanding to how each one
works. Afterwards, each of these algorithms will be implemented to show how to get the results
.For the implementation, python will be used to implement each algorithm. Then pandas and
sklearn will be the frameworks needed to analyze the datasets provided. Finally, a model
validation will be done by splitting test and training data to show which algorithm has the least
deviation and error [2].

3 – Literature review
Using machine learning techniques in energy consumption is an innovative way to conserve
energy for residential areas. Therefore, this literature review would mainly concern studies and
results from scholar articles and open-source software platforms with datasets from residential
buildings and the implementation scenario. During the research, three machine learning methods
were found to be able to predict an energy consumption forecast. Next, each model will be
implemented and validated to assess its effectiveness in predicting energy consumption.
3.1 – Method 1: Linear Regression
For predicting the energy forecast, Regression models are a popular choice for predicting energy
forecasts due to their simplicity and reasonably accurate implementation when compared to other
statistical methods.Regression is a convenient method for developing both energy and statistical
models, particularly when past data is available.[2]. Based on the formula below, linear
regression makes use of single independent variables to predict the value of a dependent
variable. So, the first step would be establishing variables for the regression models . A common
way of establishing a correlation between variables is determining a “best fit” regression
equation to go through the dataset consisting of multiple independent variables and a dependent
variable [2]. The Formula below is a simple regression formula. Y is the dependent variable, β0
is the intercept and x are the independent variables.
y = β0 + β1x1 [3].

Then, the random error is calculated using the least square function.

[2].

Where L is the least square function. The equation in the squared brackets represents the distance
of the fitted line from point i.
3.2 – Method 2: Support Vector Regression
Support vector regression is a type of machine learning regression that works on the principle of
structural risk minimization and least maximum error threshold. It aims to keep the maximum
error from a data point within a particular range or desired value [2]. The support vector
regression will be used for energy models for large institutions such as Manufacturing firms.
Below is the illustration for a support vector regression [2].

Fig 1: Support vector regression underlying optimization model [2]
Figure 1 illustrates how support vector regression finds a hyperplane that separates two classes
and classifies new points based on whether they lie on the positive or negative side of the
hyperplane [6]. The decision boundaries are +ε and -ε , while y represents the hyperplane. The
equations for the hyperplane is given as
y =wx + b [2].

Therefore, any hyperplane should satisfy this equation below.

-ε < y - wx + b < +ε [6].
3.3 – Method 3: Polynomial Regression
Polynomial Regression is a regression algorithm that models dependent variable (y) and
independent variable (x) as nth degree polynomials [3]. The equation for the Polynomial
Regression is given below:

y = β0 + β1x1 + β2x2 +..... + βnxn [2].

It is also called the special case of Multiple Linear Regression in ML. Because we add some
polynomial terms to the Multiple Linear regression equation to convert it into Polynomial
Regression [3]. The multiple linear regression uses the same formula as the linear regression, but
it is raised to the nth power. When applying this is linear while the polynomial regression is a
curved graph. The figure below shows how distinct they are.

Fig 2: The difference between linear and polynomial regression [3]
3.4 – Model Validation
Model validation helps ensure that the model can perform well when given new data. If model
validation is not performed then it is uncertain that the training model will work well when given
new data [4]. Overall, the model validation is done to ensure that the models can predict data
under given conditions. The various techniques used for validation are:
1. Hold out validation.
2. K-fold Cross-Validation.
3. Stratified K-fold Cross-Validation.
4. Leave One Out Cross-Validation.
5. Repeated Random Test-Train Splits.

All these validation techniques are recommended from data scientists. For this report ,
Regression models will be validated. The validation model that will be used is Repeated Random
test-train splits.
3.4.1 Repeated Random test-train splits
The train-test split technique is used for regression models to test the machine learning
algorithms [5]. It is implemented by using Python and Scikit learning, and the dataset is split into
two subsets.
1. Training dataset: Percentage of the data set is split here to train the algorithm and fit into
the machine learning model.
2. Test dataset: Percentage of the data set is split here using input from the test dataset to
make predictions.

3.4.2 Performance test
Finally, the performance of each model will be evaluated using R score, Mean Absolute Error
and Mean Error Squared Root. The model with the lowest R score, MAE and MSE will be
selected as the best model for this project. The energy models are going to be developed using
python programming.
4 – Implementation
The system implementation schedule would be composed of four phases.
Phase 1 (Dataset collection): The data collected from different sources to be able to predict
energy consumption in residential buildings.
Phase 2 (Data Preprocessing): This phase involves data being cleaned, formatted and filtered to
remove any inconsistencies, outliers and missing data.
Phase 3 (Model Development): Here the three machine learning algorithms will be implemented
using Python programming. Pandas and Scikit-learn will be used to analyze the data and develop
the model.
Phase 4 (Model evaluation): Here each model will undergo a test-train split to ensure that the
model is predictive. Also a performance test will be done to check the best model.
4.1–Dataset collection and Data Preprocessing
The dataset was obtained from the government of Canada. The data contains the energy
consumption of households in different Provinces of Canada [5]. The code below shows data
preprocessing.

Fig 3: Imported dataset

Fig 4: Data dropped

Fig 5: Energy type and location filtered

Fig 6: The dependent and independent variables selected

Fig 7: More independent variables are selected

Fig 8: Independent variables joined

Fig 9: The dependent variable
The given figures show a data preparation process in python using the pandas library. It
starts with importing necessary libraries and reading the dataset using pandas. Then,
unnecessary columns in the dataset are dropped using the ‘drop’ function. After that, the
data is filtered by provinces using the ‘GEO’ column, and rows with the location
‘Canada’ are dropped from the dataset. Next, the dataset is further filtered to select only
the energy type ‘Electricity’ in the ‘Energy type’ column.
Furthermore, figure 6 shows that the independent and dependent variables are selected
for the model, where the dependent variable is the total electricity consumed and the
independent variables are the number of houses using electricity, proportion of electricity
used, and the electricity used per household. The data related to each variable is selected
from the energy consumption column, and the index for all the variables is dropped for
efficient joining of data frames.
Lastly, the independent variables are concatenated to join their data frames into a singular
data frame with the same index, and the independent and dependent variables are ready
for the model development. Overall, the text provides a clear and concise overview of the
data preparation process for the given dataset in python.

4.2– Linear Regression model

Fig 10: Prediction model for linear regression
The figure 10 shows a predictive model for linear regression. The model involves one
independent variable (electricity consumed per household) and one dependent variable (total
electricity consumed). To evaluate the performance of the model, the available data is divided
into two sets using the test train split. The split used in this model is 0.25, meaning that about 31
data points are used for testing, while the remaining data points are used for training. To ensure
that the test train split is consistent and reproducible, a random state of 0 is used. Once the model
is trained on training data, it can be used to predict the values of the dependent variables (y)
based on the values of the independent variable (x) in the testing set. The x test data is passed
through the predict model, and the resulting predicted values of y are compared to actual values
in the testing set to evaluate the performance of the model.

4.2.1– Model Evaluation

Fig 11: The evaluated model for linear regression

The figure 11 computes the evaluation metrics for the model. The r2_score is the R-squared
coefficient. It ranges between 0 and 1, a 0 value means the model is a bad fit while a 1 value
shows that the model is a good fit. The R2 score is 0.161 so the model is not a good fit. Next, the
mean absolute error (MAE) measures the average absolute difference between the actual and
predicted values of a target variable. The MAE in the figure above is about 21393517.3. That is a
huge value, this means the model does not perform well.The mean squared error (MSE)
measures the average squared difference between the actual and predicted values of the target
variable. The MSE of the model is 1450010208457677.8. The number is very high therefore, the
model does not perform well.
Overall, these evaluation metrics suggest that the linear regression model is not a good fit for the
data and does not perform well in predicting the dependent variable based on the independent
variable.
4.3– Support Vector Regression model

Fig 12: Scaled values for support vector regression model

Fig 13: Prediction model for support vector regression model
The figures above show the steps taken to train a regression model on the data. The data is first
split into training and testing sets using train test split with a random state 0 and a test size of
0.25. To improve performance of the model, the variables are scaled using the standardscaler
function, which helps to standardize the range of values across the features [7]. Additionally, a
kernel function in the regression model captures non-linear relationships between the variables.
The kernel function computes the distance between data points, which is affected by the scale of
input variables. The scaled variables are then fit into the regressor model and passed through an
inverse transform to give the predicted response in figure 13. The inverse transform helps obtain
the actual values of the target variable that can be compared with actual values ‘y_test’ during
model evaluation.
4.3.1– Model Evaluation

Fig 14: The evaluated model for support vector regression
Figure 14 is the evaluation metrics for the model. The R2 score is 0.663 so the model is not a bad
fit but it is also not a very good fit. The model fit is average. Next, the MAE in the figure above
is about 10961276.05. That is a huge value however, due to the magnitude of the target values
and test values the number shows that the model performs okay. The MSE of the model is
581733323034376.6. The number is very high but due to the test values and target values being
high the model performs okay.

4.4–Polynomial Regression model

Fig 15: The predicted response for polynomial regression
For the polynomial regression model shown in figure 15, two independent variables and one
dependent variable were used. The two independent variables (X) are the number of households
that use electricity and the electricity per household. The dependent variable (Y) was the total
electricity consumed. The polynomial degree for this model is 2, and its equation can be
expressed as y = β0 + β1x1 + β2x2. The test train split is used to train the model. The split has a
random state of 0 and a test size of 0.25. The train data is transformed by the polynomial
regression and fit into the linear regression model. The test data is passed through the model
predict to give the predicted response (y).
4.4.1– Model Evaluation

Fig 16: The evaluated model for polynomial regression

Figure 16 shows the evaluation metrics for the model. The R2 score is 0.905 so the model is
definitely a good fit. Next, the MAE in the figure above is about 6555100.6. That is a huge value
however, due to the magnitude of the target values and test values the number shows that the
model performs well. The MSE of the model is 164818238851229.0. The number is very high
but due to the test values and target values being high the model performs well.

5– Conclusion and Recommendations
In conclusion, this paper has presented the development of machine learning models to predict
energy consumption in residential buildings by using three algorithms: linear regression, support
vector regression and polynomial regression. The dataset was collected from the government of
Canada and preprocessed using python’s Pandas library to remove any inconsistencies, outliers
and missing data.
The models were then developed and evaluated using evaluation metrics such as r-squared
coefficient, mean absolute error, and mean squared error. The results showed that the polynomial
regression model outperformed the other two models in terms of its prediction accuracy.
The findings of this study have significant implications for energy conservation and
sustainability efforts in residential buildings. By accurately predicting energy consumption,
building owners and operators can make informed decisions about energy usage and improve the
efficiency of their buildings, ultimately reducing energy costs and carbon emissions.
Future work may involve the use of more advanced machine learning algorithms, such as deep
learning, to improve the accuracy of energy consumption predictions. Additionally, the inclusion
of weather data may further improve the model's accuracy as weather is known to have a
significant impact on energy consumption in buildings.
Overall, this study has demonstrated the potential of machine learning models in predicting
energy consumption in residential buildings, and their potential to contribute to a more
sustainable and energy efficient future.

6 – References
[1]“Energy efficiency is finally turning a corner as the world tackles the energy and climate crises,” World
Economic Forum. [Online]. Available:
https://www.weforum.org/agenda/2023/01/energy-efficiency-energy-and-climate-crises/. [Accessed:
16-Feb-2023].
[2]C. R. Madhusudanan, “A machine learning framework for energy consumption ... - tigerprints,”
tigerprints.clemson.edu, Aug-2019. [Online]. Available:
https://tigerprints.clemson.edu/cgi/viewcontent.cgi?article=4191&context=all_theses. [Accessed:
16-Feb-2023].
[3]“Machine learning polynomial regression - javatpoint,” www.javatpoint.com. [Online]. Available:
https://www.javatpoint.com/machine-learning-polynomial-regression. [Accessed: 16-Feb-2023].
[4]D. Singh, “Deepika Singh,” Pluralsight, 06-June-2019. [Online]. Available:
https://www.pluralsight.com/guides/validating-machine-learning-models-scikit-learn. [Accessed:
16-Feb-2023].
[5]Mayuresh, “Train test split for Evaluating Machine Learning Algorithms: An important guide (2021),”
UNext, 24-Jan-2023. [Online]. Available: https://u-next.com/blogs/artificial-intelligence/train-test-split/.
[Accessed: 16-Feb-2023].
[6]A. Sethi, “Support vector regression in machine learning,” Analytics Vidhya, 02-Dec-2022. [Online].
Available:
https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/.
[Accessed: 16-Feb-2023].
[7] B. Naibei, “Getting started with support vector regression in python,” Section. [Online]. Available:
https://www.section.io/engineering-education/support-vector-regression-in-python/. [Accessed:
15-Mar-2023].
