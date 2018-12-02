# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# get_trends.R automates downloading and concatenation of 90-day daily Google Trends data


library(lubridate)
library(gtrendsR)
library(rlist)
library(tidyverse)



date_range <- "2008-01-01 2017-12-31"

get_dates_list <- function(date_range, dates_sep = months(3), date_overlap = months(1)){
  
  # Breaks date range into two dates and converts them to date format
  dates <- str_split(date_range, " ")[[1]] %>% ymd()
  
  # Initializes a list of dates and first date
  temp_date <- dates[1]
  dates_list <- list()
  
  # Appends to list of dates until past date range provided
  repeat{
    interval_start <- temp_date
    interval_end <- temp_date %m+% dates_sep
    interval_str <- paste(interval_start, interval_end)
    
    dates_list <- append(dates_list, list(interval_str, interval_start, interval_end))
    temp_date <- temp_date %m+% dates_sep %m-% date_overlap 
    
    if (temp_date >= dates[2]) {
      break
    }
  }
  
  # Returns list of dates
  dates_list
  
}

get_dates_list(date_range)




concat_trends <- function(dates_list, terms, locations){
  
  trends <- 
    gtrends(keyword = terms, geo = locations, time = dates_list[[1]][[1]])[[1]] %>% 
    as.tibble() %>% 
    mutate(.id = 1,
           date_range = dates_list[[i]][[1]])
  
  for (i in 2:length(dates_list)){
    
    trend <- 
      gtrends(keyword = terms, geo = locations, time = dates_list[[i]][[1]])[[1]] %>% 
      as.tibble() %>% 
      mutate(.id = i,
             date_range = dates_list[[i]][[1]])
    
    mean_old <-
      trends %>% 
      filter(date > dates_list[[i]][[2]] & date < dates_list[[i-1]][[3]]) %>% 
      summarize(mean = mean(hits)) %>% 
      pull()
      
    mean_new <-
      trend %>% 
      filter(date > dates_list[[i]][[2]] & date < dates_list[[i-1]][[3]]) %>% 
      summarize(mean = mean(hits)) %>% 
      pull()
      
    if (mean_old > mean_new){
      trends <-
        trends %>%
        mutate(hits = hits*mean_new/mean_old)
    } else if (mean_old < mean_new) {
      trend <-
        trend %>%
        mutate(hits = hits*mean_old/mean_new)
    }
    
    trends <-
      trends %>% 
      bind_rows(trend)
    
    print(paste(round(100*i/length(dates_list), 1), "% complete!", sep = ""))
    
  } 
  
  trends
  
}


trends <- concat_trends(dates_list, terms, locations)

trends %>% 
  ggplot(aes(x = date, y = hits, color = as.factor(.id))) +
  geom_line() +
  ylim(0, 100) + 
  theme_bw() +
  theme(aspect.ratio = 1/3) + 
  guides(color = FALSE) 





terms <- c("depression")
locations <- c("US-NY-501")

trends <- gtrends(keyword = terms, geo = locations, time = date_range)[[1]]

ggplot() + 
  geom_line(data = trend12, aes(y = hits, x = date, color = source), size = 1) + 
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



trend12 <- bind_rows(trends1, trends2, .id = "source")
trend12_adj <- bind_rows(trends1, trends2_adj, .id = "source")






