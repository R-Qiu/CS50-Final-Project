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

file <- "trends_data/nyc_depression_gtrends_2008-2017.csv"
outfile <- "trends_data/nyc_mvavg_depression_gtrends_2008-2017.csv"
# file <- "trends_data/boston_depression_gtrends_2008-2017.csv"


weights <- c(1.4, 2.4, 3.4, 4, 3.4, 2.4, 1.4)

trends <-
  read_csv(file)

trends_mvavg <- 
  trends %>% 
  mutate(moving_average = roll_mean(hits, 7, align = "center", fill = 0, weights = weights))

ggplot() +
  geom_line(data = trends, aes(x = date, y = hits), color = "black") +
  geom_line(data = trends_mvavg, aes(x = date, y = moving_average), color = "firebrick") + 
  



write_csv(trends_mvavg, outfile)

