# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data

# Import necessary packages
library(lubridate)
library(gtrendsR)
library(rlist)
library(RcppRoll)
library(tidyverse)

# Define the input and output file paths 
infile <- "trends_data/boston_depression_gtrends_2008-2017.csv"
outfile <- "trends_data/boston_mvavg_depression_gtrends_2008-2017.csv"

# Define the weights for the rolling 7-day averaging window
# This roughly follows a normal distribution
weights <- c(1.4, 2.4, 3.4, 4, 3.4, 2.4, 1.4)

# Read in the input file
trends <-
  read_csv(infile)

# Calculate a new column with the rolling mean
# Uses a 7-day rolling window centered on the day for which the rolling mean is being calculated
# Weights from above, fills values for which the window does not fit with 0
trends_mvavg <- 
  trends %>% 
  mutate(moving_average = roll_mean(hits, 7, align = "center", fill = 0, weights = weights)) %>% 
  filter(moving_average != 0)

# For visualization methods, plots both the old raw data and the new smoothed data
ggplot() +
  geom_line(data = trends, aes(x = date, y = hits), color = "black", size = 1) +
  geom_line(data = trends_mvavg, aes(x = date, y = moving_average), color = "red", size = 1) + 
  ylim(0, 100) + 
  theme_bw(base_size = 12) +
  theme(aspect.ratio = 1/3) +
  scale_colour_manual(values = c("black", "red")) +
  guides(colour = guide_legend(override.aes = list(shape = c(16, 8))))

# Writes out to a CSV file at the specified file path
write_csv(trends_mvavg, outfile)

