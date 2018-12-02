# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data


library(lubridate)
library(gtrendsR)
library(tidyverse)


terms <- c("united states")
locations <- c("US-NY", "US-RI")
dates <- "2008-01-01 2017-12-31"

trends <- gtrends(keyword = terms, geo = locations, time = dates)[[1]]

trends %>% 
  ggplot(aes(y = hits, x = date, color = geo)) + 
  geom_line(size = 1) + 
  ylim(0, 100) + 
  theme_bw() +
  theme(aspect.ratio = 1/3)
