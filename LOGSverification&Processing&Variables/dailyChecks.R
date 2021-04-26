
#clear workspace
rm(list =ls())

# setting the working directory (where to get the data from?)
#setwd("G:/VPHI/Welfare/2- Research Projects/Laura C-Klara G/_dailycheckingSystem")
setwd("//nas-vetsuisse/vetsuisse/Gruppen/VPHI/Welfare/2- Research Projects/Laura C-Klara G/_dailycheckingSystem")

#loading data
log_3_5 <- read.csv(paste0(getwd(),"/checkFiles/log_3-5.csv"), header = FALSE, sep = ';')
log_10_12 <- read.csv(paste0(getwd(),"/checkFiles/log_10-12.csv"), header = FALSE, sep = ';')
dev_3_5 <- read.csv(paste0(getwd(),"/checkFiles/dev_3-5.csv"), header = FALSE, sep = ';')
dev_10_12 <- read.csv(paste0(getwd(),"/checkFiles/dev_10-12.csv"), header = FALSE, sep = ';')

# if something changes with tags enter this in the following file and change its name to the date
# then change the date here:
id_List <- read.csv("tagID_backpack_Legring_pen_2020_06_05.csv", header = TRUE, sep = ';')

#voltage thresholds
#Threshold for minimarker and wintergarden battery change
thresh_dev = 3100
#Threshold for tag battery change
thresh_tag = 2910



###############################################################################################
#naming the variables
colnames(log_3_5) <- c('stamp', 'id','tag',  'zone', 'db', 'pen', 'zone2')
colnames(log_10_12) <- c('stamp', 'id','tag',  'zone', 'db', 'pen', 'zone2')
colnames(dev_3_5) <- c('stamp','sort','code', 'sender', 'voltage', 'rssi', 'lastReg')
colnames(dev_10_12) <- c('stamp','sort','code', 'sender', 'voltage', 'rssi', 'lastReg')


# transform time in POSIX data format (format to deal with times)
log_3_5$stamp =  as.POSIXct(log_3_5$stamp, format = '%d.%m.%Y %H:%M:%OS')
log_10_12$stamp =  as.POSIXct(log_10_12$stamp, format = '%d.%m.%Y %H:%M:%OS')
dev_3_5$stamp =  as.POSIXct(dev_3_5$stamp, format = '%d.%m.%Y %H:%M:%OS')
dev_10_12$stamp =  as.POSIXct(dev_10_12$stamp, format = '%d.%m.%Y %H:%M:%OS')
dev_3_5$lastReg =  as.POSIXct(dev_3_5$lastReg, format = '%d.%m.%Y %H:%M:%OS')
dev_10_12$lastReg =  as.POSIXct(dev_10_12$lastReg, format = '%d.%m.%Y %H:%M:%OS')

#save time from right now from computer
timeNow = Sys.time()

#save all senders registered in the device updates
sender_3_5 = unique(dev_3_5$sender)
sender_10_12 = unique(dev_10_12$sender)

#saving all tags in use
tags_3_5 = id_List$TrackingTag[id_List$Pen == 3 | id_List$Pen == 4 | id_List$Pen == 5]
tags_10_12 = id_List$TrackingTag[id_List$Pen == 10 | id_List$Pen == 11 | id_List$Pen == 12]

#saving all devices in use
dev_3_5_List = as.character(unique(dev_3_5$sender)[2:9])
dev_10_12_List = as.character(unique(dev_10_12$sender)[2:9])


#first printout line
#\n means writing will be continued in the next line 
cat('___________________________________________________________________________________________________________________________\n')
cat(paste("This is the check for the", timeNow), '\n')
cat('\n')
cat('----------------------------------------------------\n')

#################################################################
#### DEVICE UPDATE CHECKS ####


#checking last entry to make sure everything is still working
lastEntry3_5 = tail(dev_3_5[dev_3_5$sort == 'Reader',],1)
lastEntry10_12 = tail(dev_10_12[dev_10_12$sort == 'Reader',],1)

cat(paste("The last read out from the reader of 3-5 arrived at:", lastEntry3_5$lastReg), '\n')
cat(paste("The last read out from the reader of 10-12 arrived at:", lastEntry10_12$lastReg), '\n')
cat('\n')
cat('----------------------------------------------------\n')



#checking last update for all devices
cat("Devices with updates not agreeing with the last update: \n")
cat('\n')
for (sender in sender_3_5){
  
    lastUpdate = max(dev_3_5$lastReg[dev_3_5$sender == sender])
    
    #was the last update sent within the last 10 minutes?
    if (difftime(lastEntry3_5$lastReg, lastUpdate, units = "secs") > 600) {
      
      if (is.element(sender, tags_3_5)){
        
        cat(paste(sender, "had last update at", lastUpdate), '\n')
      
      } 
      else if (is.element(sender, dev_3_5_List)){
        
        cat(paste(sender, "had last update at", lastUpdate), '\n')
        
      }
      
      else {
        
        cat(paste(sender), '-> spare tag, not important \n') 
      }

      
    }
}
cat('\n')
#do it also for the other loop
for (sender in sender_10_12){
  
  lastUpdate = max(dev_10_12$lastReg[dev_10_12$sender == sender])
  
  if (difftime(lastEntry10_12$lastReg, lastUpdate, units = "secs") > 600) {
    
    if (is.element(sender, tags_10_12)){
      
      cat(paste(sender, "had last update at", lastUpdate), '\n')
      
    } else if (is.element(sender, dev_10_12_List)){
      
      cat(paste(sender, "had last update at", lastUpdate), '\n')
      
    } 
    else {
      
      cat(paste(sender), '-> spare tag, not important \n') 
    }
    
  }
}

cat('\n')
cat('----------------------------------------------------\n')

###############
#checking for voltages 

cat("Devices with critical voltages:", '\n')
cat('\n')
for (sender in sender_3_5){
  
  voltages = dev_3_5$voltage[dev_3_5$sender == sender]
  
  #save the last 5 entries for the current tag/device
  voltages5 = voltages[(length(voltages)-4): length(voltages)]
  
  if (any(is.na(voltages5))){
    
    if (sender == ""){
      
      next 
    
    } else {
      
      if (is.element(sender, tags_3_5)){
        
        cat(paste(sender, "did not register any voltage."), '\n')
      
      } 
      else if (is.element(sender, dev_3_5_List)){
        
        cat(paste(sender, "did not register any voltage."), '\n')
        
      }else {
        
        cat(paste(sender), '-> spare tag, not important \n')
        
      }
    
    }

    
  } 
  else if (any(voltages5 == 0)){
    
    next
    
  } 
  # check whether at least three of the last entries were below the critical threshold 
  else if ((sum(voltages5 < thresh_tag) > 2) & is.element(sender, tags_3_5)) {
      
      cat(paste(sender, "had min V:", min(voltages5), "and max V:", max(voltages5), "in the last 5 entries"), '\n')
      
    } 
  else if ((sum(voltages5 < thresh_dev) > 2) & is.element(sender, dev_3_5_List)){
      
      cat(paste(sender, "had min V:", min(voltages5), "and max V:", max(voltages5), "in the last 5 entries"), '\n')
      
    }
  else if (!is.element(sender, tags_3_5) & !is.element(sender, dev_3_5_List)){
      
      cat(paste(sender), '-> spare tag, not important \n')
      
  }
  else {
    next
  }
    
}

cat('\n')

# repeat all for the other loop 
for (sender in sender_10_12){
  
  voltages = dev_10_12$voltage[dev_10_12$sender == sender]
  
  voltages5 = voltages[(length(voltages)-4): length(voltages)]
  
  if (any(is.na(voltages5))){
    
    if (sender == ""){
      
      next 
      
    } else {
      
      if (is.element(sender, tags_10_12)){
        
        cat(paste(sender, "did not register any voltage."), '\n')
        
      } else if (is.element(sender, dev_10_12_List)){
        
        cat(paste(sender, "did not register any voltage."), '\n')
        
      }else {
        
        cat(paste(sender), '-> spare tag, not important \n')
        
      }
      
    }
    
  } else if (any(voltages5 == 0)){
    
    next
  
  } else if ((sum(voltages5 < thresh_tag) > 2) & is.element(sender, tags_10_12)){
      
      cat(paste(sender, "had min V:", min(voltages5), "and max V:", max(voltages5), "in the last 5 entries"), '\n')
      
    } 
    else if ((sum(voltages5 < thresh_dev) > 2) & is.element(sender, dev_3_5_List)){
      
      cat(paste(sender, "had min V:", min(voltages5), "and max V:", max(voltages5), "in the last 5 entries"), '\n')
      
    }
  else if (!is.element(sender, tags_10_12) & !is.element(sender, dev_10_12_List)){
    
    cat(paste(sender), '-> spare tag, not important \n')
    
  }
  else {
    next
  }
    
}

cat('\n')
cat('----------------------------------------------------\n')


##################################################################
#### TAG UPDATE CHECKS ####
#checking whether all tags still send transitions

cat("Tags which did not log any transitions:", '\n')
cat('\n')
for (tag in tags_3_5){
  
  #save all transitions logged by the current tag
  transitions = log_3_5[log_3_5$tag == tag,]
  
  if (nrow(transitions) == 0) {
    
    cat(paste(tag, "did not log any transitions at least since", log_3_5$stamp[1]), '\n')
    
  }
}

cat('\n')

#repeat for other loop
for (tag in tags_10_12){
  
  transitions = log_10_12[log_10_12$tag == tag,]
  
  if (nrow(transitions) == 0) {
    
    cat(paste(tag, "did not log any transitions at least since", log_10_12$stamp[1]), '\n')
    
  }
}

cat('\n')