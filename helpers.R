# CS50 Final Project
# Richard Qiu & Todd Chutichetpong
# helpers.R defines the functions used in get_trends.R

# Takes in pair of dates, outputs list of lists with overlapping date ranges
get_dates_list <- function(date_range, dates_sep=months(3), date_overlap=months(1)) {
  
  # Breaks date range into two dates and converts them to date format
  dates <- str_split(date_range, " ")[[1]] %>% ymd()
  
  # Initializes a list of dates and first date
  temp_date <- dates[1]
  dates_list <- list()
  
  # Appends to list of dates until past date range provided
  repeat {
    
    # Calculates beginning and end of interval
    interval_start <- temp_date
    interval_end <- temp_date %m+% dates_sep
    interval_str <- paste(interval_start, interval_end)
    
    dates_list <- list.append(dates_list, list(interval_str, interval_start, interval_end))
    temp_date <- temp_date %m+% dates_sep %m-% date_overlap 
    
    # Breaks out of loop if temp_date has passed the upper date bound
    if (temp_date >= dates[2]) {
      break
    }
  }
  
  # Returns list of list of dates
  dates_list
  
}


# Concatenates Google trend data with overlap, scaling appropriately
concat_trends <- function(dates_list, terms, locations, rescale_robin_williams=FALSE) {
  
  # Initializes trends with the first date interval
  trends <- 
    gtrends(keyword = terms, geo = locations, time = dates_list[[1]][[1]])[[1]] %>% 
    as.tibble()
  

  trends <-
    trends %>%
    group_by(date, geo) %>%
    summarize(hits = mean(hits)) %>%
    mutate(keywords = paste(terms, collapse=",")) %>% 
    ungroup()

  
  # Iterates over date intervals
  for (i in 2:length(dates_list)) {
    
    # Grabs trends data for specified time range, location, and search terms
    trend <- 
      gtrends(keyword = terms, geo = locations, time = dates_list[[i]][[1]])[[1]] %>% 
      as.tibble()
  
      
    trend <- 
      trend %>% 
      group_by(date, geo) %>% 
      summarize(hits = mean(hits)) %>% 
      mutate(keywords = paste(terms, collapse=";")) %>% 
      ungroup()

    
    # Calculates mean hits of older side left side of trends data for overlapping dates
    mean_old <-
      trends %>% 
      filter(date > dates_list[[i]][[2]] & date < dates_list[[i-1]][[3]]) %>% 
      summarize(mean = mean(hits)) %>% 
      pull()
    
    # Calculates mean hits of newer side left side of trends data for overlapping dates
    mean_new <-
      trend %>% 
      filter(date > dates_list[[i]][[2]] & date < dates_list[[i-1]][[3]]) %>% 
      summarize(mean = mean(hits)) %>% 
      pull()
    
    # Scales trends data by adjusting entire time period by ratio of means in overlapping dates
    if (mean_old > mean_new) {
      trends <-
        trends %>%
        mutate(hits = hits*mean_new/mean_old)
    } else if (mean_old < mean_new) {
      trend <-
        trend %>%
        mutate(hits = hits*mean_old/mean_new)
    }
    
    # Concatenates old trend data with new trend data
    trends <-
      trends %>% 
      bind_rows(trend)
    
    # Prints out progress
    print(paste(round(100*i/length(dates_list), 1), "% complete!", sep = ""))
    
  } 

  
  # Merges overlapping date regions, dropping irrelevant/temporary columns
  trends <- 
    trends %>% 
    group_by(date, keywords, geo) %>% 
    summarise(hits = mean(hits)) %>% 
    ungroup()
    
  
  # If rescale_robin_williams=TRUE, naively rescales trends by...
  # Correcting for Robin Williams-related searches the day following his death (2018-08-12)
  if (rescale_robin_williams) {
    
    # Used even-weight 11-day window for rolling mean correction for 1 week before and after day of death
    trends <-
      trends %>% 
      mutate(moving_average = roll_mean(hits, 13, align = "center", fill = 0)) %>% 
      mutate(hits = case_when(date > ymd("2014-08-04") & date < ymd("2014-08-19") ~ moving_average,
                              TRUE ~ hits)) %>% 
      select(-moving_average)
    

    # Rescale trends to 100
    trends_max <-
      trends %>%
      summarize(max = max(hits)) %>% 
      pull()
    
    trends <-
      trends %>% 
      mutate(hits = hits*100/trends_max)

    
  } 
    
  # Return trends tibble
  trends
    
  
}
