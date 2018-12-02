# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data

# Import necessary packages
library(lubridate)
library(gtrendsR)
library(rlist)
library(tidyverse)

# Import functions from helpers.R
source("helpers.R")


# Define Google Trends parameters
terms <- c("depression")
locations <- c("US-NY-501")
date_range <- "2008-01-01 2017-12-31"

# Define output filename
filename <- "NYC_gtrends_2008-2017.csv"

# Get list of date intervals
dates_list <- get_dates_list(date_range)

# Get Google Trends 
trends <- concat_trends(dates_list, terms, locations, rescale_robin_williams=TRUE)

# Writes out CSV file of Google Trends data
write_csv(trends, filename)











