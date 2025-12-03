setwd("__your_path_here__") 

rm(list=ls())

library(rgdal)
library(sp)
library(sf)
library(maptools)
library(tmap)
library(foreign)
library(dplyr)
library(gridExtra)
library(rstudioapi) 
library(ggplot2)

set_wd <- function() {
  current_path <- getActiveDocumentContext()$path 
  setwd(dirname(current_path ))
  print( getwd() )
}

#Loading datasets
eu <- readOGR(dsn = getwd(), layer ="NUTS_RG_03M_2013_4326_LEVL_1")
eu <- eu[eu$CNTR_CODE=="DE", ]
#writeOGR(eu,".","DE",driver="ESRI Shapefile")
eu <- readOGR(".","DE")
camps <- read.csv("camp_coords.csv")
campsDE <- camps[ which(camps$country=='DE'), ]
tif_sf <- st_as_sf(campsDE, coords = c("long", "lat"))
kreise <- readOGR(dsn = getwd(), layer ="currentmap_interpolated")
count <- read.dta("count.dta")
kreiseq <- merge(kreise,count,by="newid")
de1925 <-readOGR(dsn = getwd(), layer ="figure1") 
state <-read.dta("state1925.dta")
de1925q <- merge(de1925,state,by="id")


# Merge polygons by ID
state.coords <- coordinates(de1925)
de1925.union <- unionSpatialPolygons(de1925, de1925q$oldid)


#Map construction (current)
#map <- tm_shape(kreiseq) +
#  tm_fill("count") +
#  tm_borders()+
#  tm_shape(eu) +
#  tm_borders(col = "blue", lwd = 2)+
#  tm_shape(tif_sf)+ 
#  tm_dots("red", size=0.5, border.col = "black", border.lwd=1, size.lim = c(0, 11e6))

#Save map
#tmap_save(map, "map_current_n.pdf", width=1920, height=1080, asp=0,tmap_options(check.and.fix = TRUE))


#Map construction (1925)
#map2 <- tm_shape(kreiseq) +
#  tm_fill("count") +
#  tm_borders()+
#  tm_shape(de1925.union) +
#  tm_borders(col = "blue", lwd = 2)+
#  tm_shape(tif_sf)+ 
#  tm_dots("red", size=0.5, border.col = "black", border.lwd=1, size.lim = c(0, 11e6))


#Save map
#tmap_save(map2, "map_1925.pdf", width=1920, height=1080, asp=0,tmap_options(check.and.fix = TRUE))

eu$NUTS_NAME[eu$NUTS_NAME=="MECKLENBURG-VORPOMMERN"] <- "MECKLENBURG-\nVORPOMMERN"
eu$NUTS_NAME[eu$NUTS_NAME=="NORDRHEIN-WESTFALEN"] <- "NORDRHEIN-\nWESTFALEN"
eu$NUTS_NAME[eu$NUTS_NAME=="BADEN-WÜRTTEMBERG"] <- "BADEN-\nWÜRTTEMBERG"
eu$NUTS_NAME[eu$NUTS_NAME=="RHEINLAND-PFALZ"] <- "RHEINLAND-\nPFALZ"
eu$NUTS_NAME[eu$NUTS_NAME=="SACHSEN-ANHALT"] <- "SACHSEN-\nANHALT"
eu$NUTS_NAME[eu$NUTS_NAME=="SCHLESWIG-HOLSTEIN"] <- "SCHLESWIG-\nHOLSTEIN"
eu$NUTS_NAME[eu$NUTS_NAME=="BRANDENBURG"] <- "\n\n\nBRANDENBURG"
eu$NUTS_NAME[eu$NUTS_NAME=="NIEDERSACHSEN"] <- "\n\nNIEDERSACHSEN"

#Colored map with 1925 overlay
map3 <- tm_shape(eu) +
  tm_polygons("NUTS_NAME", style = "cat", palette="Accent", alpha = 1, legend.show = FALSE)+ 
  tm_text("NUTS_NAME", scale=0.7)+
  tm_shape(de1925.union) +
  tm_borders(col = "black", lwd = 1)+
  tm_shape(tif_sf)+
  tm_bubbles(col="darkred",size=0.5)


#Save map
#tmap_save(map3, "stateborders.pdf", width=1920, height=1080, asp=0,tmap_options(check.and.fix = TRUE))
tmap_save(map3, "figure3.pdf")