# CSE523-Machine-Learning-Code-Warriors

## Team Members
- **Vishvas Patel**
- **Deep Patel**
- **Dhruv Panchal**
- **Devarsh Suthar**

# 1. Introduction

Predicting the condition of the atmosphere using science and technology is known as rainfall prediction. We compared all the classifiers mentioned below and  analysed the results. The graphs for the same can be found below in the results section.

We extracted data of Ahmedabad city from the past 11 years (i.e, 2009-2020 ) from an API [2]. The extracted data includes hourly forecasts of 19 parameters including   precipitation(MM), temperature, dew point, wind speed, pressure, visibility, etc.

We first applied the random forest classification algorithm to our dataset and analyzed the accuracy score.
Similarly, we applied the Support Vector Classification (SVC) algorithm to our dataset and analyzed the accuracy score.

For the feature selection, we used the Random forest classifier with permutation importance to achieve the most important features. After that, we trained our model with the selected important features and analyzed the accuracy score. 
Similarly, we analyzed accuracy of  the support vector classifier (SVC) with the permutation importance feature selection.

As an example, for the random forest classifier, we obtained important features such as DewPointC, cloudcover, uvIndex, and visibility. 
Similarly, we used the Random forest classifier and svc with chi-square test to find the k best features. After that, we trained our model with the selected important features and analyzed the accuracy score. 
In addition, we applied logistic regression and gradient boosting classifiers.
Finally, the best classifier models are chosen by comparing the trained models for two (Rain and no rain) and four classifications (No rain, Drizzle, Moderate rain, and Heavy rain).
 
 


# 2. Dataset extract using API

# *Install this dependency*
pip install wwo-hist

# *Code To generate dataset*
```
from wwo_hist import retrieve_hist_data
import os
os.chdir("./PATH")

frequency = 24
start_date = '1-JAN-2009'
end_date = '11-4-2021'
api_key = 'Your-API' #your api key from worldweatheronline website
location_list = ['Ahmedabad']
hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)

```



# 3. Results
# Random Forest Feature Selection
![RandomForestFeatureSelection](https://user-images.githubusercontent.com/60318509/114301540-f47c3800-9ae2-11eb-8e86-cdc9a54c2a8d.JPG)
# SVC Feature Selection
![SVC_FeatureSelection](https://user-images.githubusercontent.com/60318509/114301541-f5ad6500-9ae2-11eb-856b-52a6d5c6d51d.JPG)
# Grddient Boosting Feature Selection
![GradientBoostingFeatureSelection](https://user-images.githubusercontent.com/60318509/114301553-f7772880-9ae2-11eb-8dc9-9a7674462e27.JPG)
# SVC with Chi Square test
![SVC_permutation_ChiSquare](https://user-images.githubusercontent.com/60318509/114301542-f645fb80-9ae2-11eb-861d-a56771986c6a.png)
# Comparission for Four classifications
![Comparison_four_rain_classifications](https://user-images.githubusercontent.com/60318509/114301544-f645fb80-9ae2-11eb-946c-b8889344310f.png)
# Comparission for two classifications
![Comparison_two_rain_classifications](https://user-images.githubusercontent.com/60318509/114301547-f6de9200-9ae2-11eb-995f-d0c046ecdce9.png)
# Gradient Boosting
![Gradient_Boosting](https://user-images.githubusercontent.com/60318509/114301550-f7772880-9ae2-11eb-9ffd-5703e815f5fd.png)
# Logistic Regression
![Logistic_Regression](https://user-images.githubusercontent.com/60318509/114301556-f80fbf00-9ae2-11eb-9162-2cea5f4d8af2.png)
# Random forest with Chi Square Test
![Random_Forest_ChiSquare](https://user-images.githubusercontent.com/60318509/114301558-f8a85580-9ae2-11eb-9765-7f32a54a52ca.png)
# Random Forest With permutation imporance
![Random_Forest_permutation_importance](https://user-images.githubusercontent.com/60318509/114301559-f940ec00-9ae2-11eb-8347-2fd070caee2a.png)

# 4. References
```
1.http://www.ijstr.org/final-print/jan2020/Prediction-Of-Rainfall-Using-Machine-Learning-Techniques.pd
2.https://towardsdatascience.com/obtain-historical-weather-forecast-data-in-csv-format-using-python-5a6c090fc828
3.https://arxiv.org/pdf/1910.13827v1.pdf
4.Thirumalai, Chandrasegar, et al. "Heuristic prediction of rainfall using machine learning techniques." 2017 International Conference on Trends in Electronics and Informatics (ICEI). IEEE, 2017.
5.Geetha, A., and G. M. Nasira. "Data mining for meteorological applications: Decision trees for modeling rainfall prediction." 2014 IEEE International Conference on Computational Intelligence and Computing Research. IEEE, 2014
```
