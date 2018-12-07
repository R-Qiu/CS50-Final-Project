# Weather Got You Down?
### Predicting Depression Trends Using High-Resolution Weather Data

## Summary

Weather data was used as inputs into a neural network model to predict depression in a certain cities (as measured by the normalized Google searches from Google Trends). Weather features that were deemed to be related to depression and mood were extracted. The Keras library was used to construct the neural network with 2 hidden layers (each with 10 neurons). 8 years worth of data were used to train the neural network, while 2 years are reserved as a blind test. The result found that there is indeed a significant association between weather and depression as the model could accurately predict depression trend with a low MSE value. 

A brief presentation which graphically depicts these methods as well as results can be viewed [here](https://goo.gl/8sSd8Q).


## Prerequisites
**R Libraries (R Version >= 3.5.0):**

* `Lubridate`

* `gtrendsR`

* `rlist`

* `Tidyverse`

* `RcppRoll`


**Python Packages (Python Version >= 3.6):**

* `Scikit-Learn`

* `Keras`

* `Pandas`

* `Numpy`

* `Tensorflow`

* `matplotlib`


**Weather data** 

"Local Climatological" datasets in CSV format, retrievable through the National Oceanographic and Atmospheric Administration's National Climatic Data Center and the Climate Data Record Program [here](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd). Additionally, some sample datasets are provided in the `weather_data/` folder in the source code (available below).


## Running the Program

### Collecting Google Trends Data
* Open `get_trends.R` and specify the following parameters:
    - Up to 5 Google search terms for which to average hits
	      - Example format: `terms <- c(“apple”, “microsoft”, “facebook”, “amazon”)`
    - Up to 5 locations (only more than one search term *or* location can be specified)
        - Example format: `locations <- c("US-NY-501", “US-CA-807”, "US-MA-506")`
        - Codes for various cities can be found in `locations.txt`
    - The date range for which to retrieve Google Trends data (up to 10 years)
        - Example format: `date_range <- "2008-01-01 2017-12-31"`
    - The file path for the resulting CSV file of Google Trends data
        - Example format: `outpath <- “trends_data/nyc_gtrends.csv"`
    - If “depression” is a search term, it is highly recommended to also set `rescale_robin_williams=TRUE` as an argument to `concat_trends()`
    	  - Example format: `trends <- concat_trends(dates_list, terms, locations, rescale_robin_williams=TRUE)`

* Run `get_trends.R` 

* (Optional) To smooth the data, open `normalize_trends.R` and specify the following parameters:
    - The file path for the CSV file with data to be smoothed
        - Example format: `infile <- "trends_data/nyc_gtrends.csv"`
    - The file path for the resulting CSV file of smoothed data
        - Example format: `outfile <- "trends_data/nyc_smooth_gtrends.csv"`

* (Optional) Run `normalize_trends.R`


### Running the Model
Feature extraction and model implementation is in the `main.py`. Run the script will run automatically extract the features as well as feed them into the neural network model. 

* Under `load dataset`, specify the file path for the csv file that will be used as input data. Under `load google trends`, specify the file path for the csv that will be use as the label (truth). 

* Once, the model is trained the loss curve will be plotted. After the matplotlib graph is closed, the trained model will automatically predict Google Trends hits using the reserved blind test and plot it for the user.

* `Compile all features into a feature dataframe` can be adjusted if user wants to drop certain features from the input data

* Neural Network Architecture can be adjusted in the `Create model` section


## Meta

### Authors

Richard Qiu, Harvard ‘22

Todd Chutichetpong, Harvard ‘22


### Source Code
[https://github.com/R-Qiu/Weather-Got-You-Down](https://github.com/R-Qiu/Weather-Got-You-Down)






