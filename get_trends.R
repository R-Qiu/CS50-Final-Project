# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data

# Import necessary packages
library(lubridate)
library(gtrendsR)
library(rlist)
library(RcppRoll)
library(tidyverse)

# Import functions from helpers.R
source("helpers.R")


# Tibble for reference and quick lookup of original cities used in this project
our_locations <-
  tibble("New York City" = "US-NY-501",
         "Chicago" = "US-IL-602",
         "Boston" = "US-MA-506",
         "San Francisco" = "US-CA-807",
         "Seattle" = "US-WA-819",
         "Los Angeles" = "US-CA-803",
         "Charlotte" = "US-NC-517",
         "Baltimore" = "US-MD-512") %>% 
  gather(city, code)


# Define Google Trends parameters
# List of locations available in trends_data/locations.txt
terms <- c("depression")
locations <- c("US-NY-501")
date_range <- "2008-01-01 2017-12-31"

# Define output path and filename
outpath <- "trends_data/nyc_depression_gtrends_2008-2017.csv"

# Get list of date intervals
dates_list <- get_dates_list(date_range)

# Get Google Trends 
trends <- concat_trends(dates_list, terms, locations, rescale_robin_williams=TRUE)

# Writes out CSV file of Google Trends data
write_csv(trends, outpath)


