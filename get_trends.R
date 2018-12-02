# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data


library(lubridate)
library(gtrendsR)
library(rlist)
library(tidyverse)



date_range <- "2008-01-01 2017-12-31"
dates_sep <- months(2)

get_dates_list <- function(date_range, dates_sep = months(2)){
  
  # Breaks date range into two dates and converts them to date format
  dates <- str_split(date_range, " ")[[1]] %>% ymd()
  
  # Initializes a list of dates and first date
  temp_date <- dates[1]
  dates_list <- list()
  
  # Appends to list of dates until past date range provided
  repeat{
    dates_list <- list.append(dates_list, paste(temp_date, temp_date %m+% dates_sep))
    temp_date <- temp_date %m+% dates_sep
    if (temp_date >= dates[2]) {
      break
    }
  }
  
  # Returns list of dates
  dates_list
  
}

get_dates_list(date_range)




terms <- c("depression")
locations <- c("US-NY-501")

trends <- gtrends(keyword = terms, geo = locations, time = date_range)[[1]]

ggplot() + 
  geom_line(data = trend12_adj, aes(y = hits, x = date, color = source), size = 1) + 
  geom_line(data = trends, aes(x = date, y = hits), size = 1, color = "#C77CFF") +
  ylim(0, 100) + 
  theme_bw() +
  theme(aspect.ratio = 1/3) + 
  guides(color = FALSE) 
  

trends1 <- gtrends(keyword = terms, geo = locations, time = "2008-01-01 2013-12-31")[[1]]
trends2 <- gtrends(keyword = terms, geo = locations, time = "2012-01-01 2017-12-31")[[1]]

mean1 <- 
  trends1 %>%
  filter(date > ymd("2012-01-01") & date < ymd("2013-12-31")) %>% 
  summarize(mean = mean(hits)) %>% 
  pull()

mean2 <- 
  trends2 %>% 
  filter(date > ymd("2012-01-01") & date < ymd("2013-12-31")) %>% 
  summarize(mean = mean(hits)) %>% 
  pull()

trends2_adj <- 
  trends2 %>% 
  mutate(hits = hits*mean1/mean2)


trend12_adj <- bind_rows(trends1, trends2_adj, .id = "source")





