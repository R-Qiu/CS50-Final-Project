# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data


library(lubridate)
library(gtrendsR)
library(rlist)
library(tidyverse)


terms <- c("united states")
locations <- c("US-NY", "US-RI")
date_range <- "2008-01-01 2017-12-31"

dates <- str_split(date_range, " ")[[1]] %>% ymd()


temp_date <- dates[1]
dates_sep <- months(2)
dates_list <- list()
repeat{
  dates_list <- list.append(dates_list, c(temp_date, temp_date %m+% dates_sep))
  temp_date <- temp_date %m+% dates_sep
  if (temp_date >= dates[2]) {
    break
  }
}


trends <- gtrends(keyword = terms, geo = locations, time = dates)[[1]]

trends %>% 
  ggplot(aes(y = hits, x = date, color = geo)) + 
  geom_line(size = 1) + 
  ylim(0, 100) + 
  theme_bw() +
  theme(aspect.ratio = 1/3)
