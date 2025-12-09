setwd("./") 

rm(list=ls())
set.seed(12435)

## packages to be added 
library(stargazer)
library(tidyverse)
library(lmtest)
library(plm)
library(dplyr)
devtools::install_github('gibbonscharlie/bfe')
library(bfe)


################# READ IN AND PREP DATA

## main EVS data

evs<-read.csv('EVS_main.csv', stringsAsFactors = F)

## naming the fixed effects
evs$f.state <- as.factor(evs$state)
evs$state <- recode(evs$state, DE1 = "WEST:\nBaden-Wurttemberg",
                    DE2 = "WEST:\nBavaria", 
                    DE3 = "EAST:\nBerlin",
                    DE4 = "EAST:\nBrandenburg",
                    DE5 = "WEST:\nBremen",
                    DE6 = "WEST:\nHamburg",
                    DE7 = "WEST:\nHessen",
                    DE8 = "EAST:\nMecklenburg-Vorpommern",
                    DE9 = "WEST:\nLower Saxony",
                    DEA = "WEST:\nNorth Rhine-Westphalia",
                    DEB = "WEST:\nRhineland Palatinate",
                    DEC = "WEST:\nSaarland",
                    DED = "EAST:\nSaxony",
                    DEE = "EAST:\nSaxony-Anhalt",
                    DEF = "WEST:\nSchleswig-Holstein",
                    DEG = "EAST:\nThuringia")

evs %>%
  group_by(state) %>%
  summarise(mean = mean(intolerance), n = n())

evs$DistanceCat <- cut(
  evs$Distance,
  breaks = quantile(evs$Distance, c(0, .2, .4, .6, .8, 1)),
  labels = c("first", "second", "third", "fourth", "fifth"),
  right  = FALSE,
  include.lowest = TRUE
)
table(evs$state,evs$DistanceCat)


## EVS with Weimar-era administrative boundaries

evs_weimar <- read.csv('evs_weimar.csv', stringsAsFactors = F)
evs_weimar$Distance <- evs_weimar$distance

# create fixed effects for weimar admistrative units (states and Prussian provinces)
evs_weimar$oldland_pruprov[evs_weimar$oldland_pruprov==2000] <- 2001
evs_weimar$weimarprov <- evs_weimar$oldland
evs_weimar$weimarprov[evs_weimar$weimarprov==1000] <- evs_weimar$oldland_pruprov[evs_weimar$weimarprov==1000]


## 2017 Election Data

elect_data<-read.csv('elections_2017.csv', stringsAsFactors = F)

# need to delete states with no internal variation in Distance for the reweighting analysis
elect_data_bfe <- subset(elect_data, NAME_1!="Berlin" & NAME_1!="Hamburg")



################# TABLE 1 and TABLE A1

# Outcome: Intolerance toward outgroups

m1b <- lm(intolerance~Distance,evs)
m1bfe <- lm(intolerance~Distance + I(state),evs)

#With Pre-treatment variables
m1<-lm(intolerance~Distance+prop_jewish25+unemployment33+population25+nazishare33 ,evs)
m1fe<-lm(intolerance~Distance+prop_jewish25+unemployment33+population25+nazishare33+I(state) ,evs)

#With Pre/Post-treatment variables
m1f<-lm(intolerance~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          #Post-treatment
          lr+
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west,evs)
#With Pre/Post-treatment variables
m1ffe<-lm(intolerance~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          I(state)+
          #Post-treatment
          lr+
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west,evs)

#G-estimator 1st stage
m1s<-lm(I(intolerance - 
              coef(m1f)['immigrants07']*immigrants07 -
              coef(m1f)['lr']*lr -
              coef(m1f)['unemp']*unemp - 
              coef(m1f)['unemployment07']*unemployment07 - 
              coef(m1f)['educ']*educ -
              coef(m1f)['urban_scale']*urban_scale) ~ 
            Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33, evs)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=6)
for(b in 1:boots){
  d.star <- evs[sample(1:nrow(evs), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(intolerance~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(intolerance - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33,
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:6]
}
#Extracting bootstrapped SEs
SEs1 <- apply(fl.boots,2,sd)


#G-estimator 1st stage
m1sfe<-lm(I(intolerance - 
            coef(m1ffe)['immigrants07']*immigrants07 -
            coef(m1ffe)['lr']*lr -
            coef(m1ffe)['unemp']*unemp - 
            coef(m1ffe)['unemployment07']*unemployment07 - 
            coef(m1ffe)['educ']*educ -
            coef(m1ffe)['urban_scale']*urban_scale) ~ 
          Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+I(state), evs)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs[sample(1:nrow(evs), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(intolerance~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     I(state)+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(intolerance - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33+
                      I(state),
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs1fe <- apply(fl.boots,2,sd)


stargazer(m1b,m1bfe,m1,m1fe,m1s,m1sfe,se=list(NULL,NULL,NULL,NULL,SEs1,SEs1fe),dep.var.labels.include = FALSE,type="text",
          out="table1_panelA.txt", no.space=T, 
          keep = c("Distance","prop_jewish25","unemployment33","population25","nazishare33"),
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Land Fixed Effects","No","Yes","No","Yes","No","Yes"),
                         c("Post-Treatment Controls","No","No","No","No","Yes","Yes"),
                         c("Method","OLS","OLS","OLS","OLS","G-est","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))



stargazer(m1bfe,m1fe,m1sfe,se=list(NULL,NULL,SEs1fe),dep.var.labels.include = FALSE,type="text",
          out="tableA1_panelA.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","OLS","OLS","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))


# Outcome: Immigrant Resentment
m2b <- lm(resentment~Distance,evs)
m2bfe <- lm(resentment~Distance + I(state),evs)

#With Pre-treatment variables
m2<-lm(resentment~Distance+prop_jewish25+unemployment33+population25+nazishare33 ,evs)
m2fe<-lm(resentment~Distance+prop_jewish25+unemployment33+population25+nazishare33+I(state) ,evs)

#With Pre/Post-treatment variables
m2f<-lm(resentment~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          #Post-treatment
          lr+
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west,evs)

#With Pre/Post-treatment variables
m2ffe<-lm(resentment~Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+
            #Post-treatment
            lr+
            immigrants07+
            unemployment07+
            unemp+
            educ+
            female+
            age+
            urban_scale+
            west+I(state) ,evs)


#G-estimator 1st stage
m2s<-lm(I(resentment - 
            coef(m2f)['immigrants07']*immigrants07 -
            coef(m2f)['lr']*lr -
            coef(m2f)['unemp']*unemp - 
            coef(m2f)['unemployment07']*unemployment07 - 
            coef(m2f)['educ']*educ -
            coef(m2f)['urban_scale']*urban_scale) ~ 
          Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33, evs)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs[sample(1:nrow(evs), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(resentment~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(resentment - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33,
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs2 <- apply(fl.boots,2,sd)


#G-estimator 1st stage
m2sfe<-lm(I(resentment - 
              coef(m2ffe)['immigrants07']*immigrants07 -
              coef(m2ffe)['lr']*lr -
              coef(m2ffe)['unemp']*unemp - 
              coef(m2ffe)['unemployment07']*unemployment07 - 
              coef(m2ffe)['educ']*educ -
              coef(m2ffe)['urban_scale']*urban_scale) ~ 
            Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+I(state), evs)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs[sample(1:nrow(evs), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(resentment~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     I(state)+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(resentment - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33+
                      I(state),
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs2fe <- apply(fl.boots,2,sd)


stargazer(m2b,m2bfe,m2,m2fe,m2s,m2sfe,se=list(NULL,NULL,NULL,NULL,SEs2,SEs2fe),dep.var.labels.include = FALSE,type="text",
          out="table1_panelB.txt", no.space=T, 
          keep = c("Distance","prop_jewish25","unemployment33","population25","nazishare33"),
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Land Fixed Effects","No","Yes","No","Yes","No","Yes"),
                         c("Post-Treatment Controls","No","No","No","No","Yes","Yes"),
                         c("Method","OLS","OLS","OLS","OLS","G-est","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))


stargazer(m2bfe,m2fe,m2sfe,se=list(NULL,NULL,SEs2fe),dep.var.labels.include = FALSE,type="text",
          out="tableA1_panelB.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","OLS","OLS","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))



# Outcome: Support for Extreme Right-Wing Parties
m3b <- lm(far_right~Distance,evs)
m3bfe <- lm(far_right~Distance + I(state),evs)

#With Pre-treatment variables
m3<-lm(far_right~Distance+prop_jewish25+unemployment33+population25+nazishare33 ,evs)
m3fe<-lm(far_right~Distance+prop_jewish25+unemployment33+population25+nazishare33+I(state) ,evs)

#With Pre/Post-treatment variables
m3f<-lm(far_right~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          #Post-treatment
          lr+ 
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west ,evs)

m3ffe<-lm(far_right~Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+
            #Post-treatment
            lr+ 
            immigrants07+
            unemployment07+
            unemp+
            educ+
            female+
            age+
            urban_scale+
            west+I(state) ,evs)



#G-estimator 1st stage
m3s<-lm(I(far_right - 
            coef(m3f)['immigrants07']*immigrants07 -
            coef(m3f)['lr']*lr -
            coef(m3f)['unemp']*unemp - 
            coef(m3f)['unemployment07']*unemployment07 - 
            coef(m3f)['educ']*educ -
            coef(m3f)['urban_scale']*urban_scale) ~ 
          Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33, evs)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs[sample(1:nrow(evs), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(far_right~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(far_right - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33,
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs3 <- apply(fl.boots,2,sd)


#G-estimator 1st stage
m3sfe<-lm(I(far_right - 
              coef(m3ffe)['immigrants07']*immigrants07 -
              coef(m3ffe)['lr']*lr -
              coef(m3ffe)['unemp']*unemp - 
              coef(m3ffe)['unemployment07']*unemployment07 - 
              coef(m3ffe)['educ']*educ -
              coef(m3ffe)['urban_scale']*urban_scale) ~ 
            Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+I(state), evs)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs[sample(1:nrow(evs), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(far_right~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     I(state)+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(far_right - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33+
                      I(state),
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs3fe <- apply(fl.boots,2,sd)



stargazer(m3b,m3bfe,m3,m3fe,m3s,m3sfe,se=list(NULL,NULL,NULL,NULL,SEs3,SEs3fe),dep.var.labels.include = FALSE,type="text",
          out="table1_panelC.txt", no.space=T, 
          keep = c("Distance","prop_jewish25","unemployment33","population25","nazishare33"),
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Land Fixed Effects","No","Yes","No","Yes","No","Yes"),
                         c("Post-Treatment Controls","No","No","No","No","Yes","Yes"),
                         c("Method","OLS","OLS","OLS","OLS","G-est","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))

stargazer(m3bfe,m3fe,m3sfe,se=list(NULL,NULL,SEs3fe),dep.var.labels.include = FALSE,type="text",
          out="tableA1_panelC.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","OLS","OLS","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))



####### TABLE A2:  HAUSMAN Tests

# bivariate models
form <- intolerance~Distance
wi1 <- plm(form, data = evs, model = "within", index="state")
re1 <- plm(form, data = evs, model = "random", index="state")
pooled1 <- plm(form, data = evs, model = "pooling", index="state")
re.v.pooled1 <- round(phtest(re1, pooled1)$p.value, digits=3)
fe.v.pooled1 <- round(phtest(wi1, pooled1)$p.value, digits=3)
fe.v.re1 <- round(phtest(wi1, re1)$p.value, digits=3)

form <- resentment~Distance
wi2 <- plm(form, data = evs, model = "within", index="state")
re2 <- plm(form, data = evs, model = "random", index="state")
pooled2 <- plm(form, data = evs, model = "pooling", index="state")
re.v.pooled2 <- round(phtest(re2, pooled2)$p.value, digits=3)
fe.v.pooled2 <- round(phtest(wi2, pooled2)$p.value, digits=3)
fe.v.re2 <- round(phtest(wi2, re2)$p.value, digits=3)

form <- far_right~Distance
wi3 <- plm(form, data = evs, model = "within", index="state")
re3 <- plm(form, data = evs, model = "random", index="state")
pooled3 <- plm(form, data = evs, model = "pooling", index="state")
re.v.pooled3 <- round(phtest(re3, pooled3)$p.value, digits=3)
fe.v.pooled3 <- round(phtest(wi3, pooled3)$p.value, digits=3)
fe.v.re3 <- round(phtest(wi3, re3)$p.value, digits=3)

stargazer(pooled1,re1,wi1, pooled2,re2,wi2, pooled3,re3,wi3, 
          out="tableA2_panelA.txt", no.space=T, keep="Distance",
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","Pooled","RE","FE","Pooled","RE","FE","Pooled","RE","FE"),
                         c("RE v Pooled",re.v.pooled1,"","",re.v.pooled2,"","",re.v.pooled3,"",""),
                         c("FE v Pooled",fe.v.pooled1,"","",fe.v.pooled2,"","",fe.v.pooled3,"",""),
                         c("FE v RE",fe.v.re1,"","",fe.v.re2,"","",fe.v.re3,"","")))

# prewar only models
form <- intolerance~Distance+prop_jewish25+unemployment33+population25+nazishare33
wi1 <- plm(form, data = evs, model = "within", index="state")
re1 <- plm(form, data = evs, model = "random", index="state")
pooled1 <- plm(form, data = evs, model = "pooling", index="state")
re.v.pooled1 <- round(phtest(re1, pooled1)$p.value, digits=3)
fe.v.pooled1 <- round(phtest(wi1, pooled1)$p.value, digits=3)
fe.v.re1 <- round(phtest(wi1, re1)$p.value, digits=3)

form <- resentment~Distance+prop_jewish25+unemployment33+population25+nazishare33
wi2 <- plm(form, data = evs, model = "within", index="state")
re2 <- plm(form, data = evs, model = "random", index="state")
pooled2 <- plm(form, data = evs, model = "pooling", index="state")
re.v.pooled2 <- round(phtest(re2, pooled2)$p.value, digits=3)
fe.v.pooled2 <- round(phtest(wi2, pooled2)$p.value, digits=3)
fe.v.re2 <- round(phtest(wi2, re2)$p.value, digits=3)

form <- far_right~Distance+prop_jewish25+unemployment33+population25+nazishare33
wi3<- plm(form, data = evs, model = "within", index="state")
re3 <- plm(form, data = evs, model = "random", index="state")
pooled3 <- plm(form, data = evs, model = "pooling", index="state")
re.v.pooled3 <- round(phtest(re3, pooled3)$p.value, digits=3)
fe.v.pooled3 <- round(phtest(wi3, pooled3)$p.value, digits=3)
fe.v.re3 <- round(phtest(wi3, re3)$p.value, digits=3)

stargazer(pooled1,re1,wi1, pooled2,re2,wi2, pooled3,re3,wi3, 
          out="tableA2_panelB.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","Pooled","RE","FE","Pooled","RE","FE","Pooled","RE","FE"),
                         c("RE v Pooled",re.v.pooled1,"","",re.v.pooled2,"","",re.v.pooled3,"",""),
                         c("FE v Pooled",fe.v.pooled1,"","",fe.v.pooled2,"","",fe.v.pooled3,"",""),
                         c("FE v RE",fe.v.re1,"","",fe.v.re2,"","",fe.v.re3,"","")))



########## TABLE A3: REWEIGHTED ANALYSES

# first prep the data. Have to exclude any state with no variation in Distance
data.evs <- data.frame(evs)
data.evs <- data.evs[data.evs$f.state!="DE3" & data.evs$f.state!="DE6",]

# estimate and save results for intolerance
m1.iwe <- EstimateIWE(y="intolerance", treatment="Distance", group="f.state", 
                      controls=c("prop_jewish25","unemployment33","population25","nazishare33"),
                      data=data.evs, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m1.fe <- rbind(m1.iwe$fe.est, sqrt(m1.iwe$fe.var), m1.iwe$fe.est/sqrt(m1.iwe$fe.var))
m1.iwe <- rbind(m1.iwe$swe.est, sqrt(m1.iwe$swe.var), m1.iwe$swe.est/sqrt(m1.iwe$swe.var))
m1.base <- rbind(summary(m1)$coefficients[2,1],summary(m1)$coefficients[2,2],summary(m1)$coefficients[2,3])

m1.rwe <- EstimateRWE(y="intolerance", treatment="Distance", group="f.state", 
                      controls=c("prop_jewish25","unemployment33","population25","nazishare33"),
                      data=data.evs, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m1.rwe <- rbind(m1.rwe$swe.est, sqrt(m1.rwe$swe.var), m1.rwe$swe.est/sqrt(m1.rwe$swe.var))

res.m1 <- cbind(m1.base, m1.fe, m1.iwe, m1.rwe)
colnames(res.m1) <- c("Pooled","FE","IWE","RWE")
write.table(res.m1, file="tableA3_panelA.txt", row.names=FALSE)

stargazer(m1,m1fe,dep.var.labels.include = FALSE,type="text",
          out="tableA3_panelA_columns1and2.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))


# estimate and save results for resentment
m2.iwe <- EstimateIWE(y="resentment", treatment="Distance", group="f.state", 
                      controls=c("prop_jewish25","unemployment33","population25","nazishare33"),
                      data=data.evs, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m2.fe <- rbind(m2.iwe$fe.est, sqrt(m2.iwe$fe.var), m2.iwe$fe.est/sqrt(m2.iwe$fe.var))
m2.iwe <- rbind(m2.iwe$swe.est, sqrt(m2.iwe$swe.var), m2.iwe$swe.est/sqrt(m2.iwe$swe.var))
m2.base <- rbind(summary(m2)$coefficients[2,1],summary(m2)$coefficients[2,2],summary(m2)$coefficients[2,3])

m2.rwe <- EstimateRWE(y="resentment", treatment="Distance", group="f.state", 
                      controls=c("prop_jewish25","unemployment33","population25","nazishare33"),
                      data=data.evs, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m2.rwe <- rbind(m2.rwe$swe.est, sqrt(m2.rwe$swe.var), m2.rwe$swe.est/sqrt(m2.rwe$swe.var))

res.m2 <- cbind(m2.base, m2.fe, m2.iwe, m2.rwe)
colnames(res.m2) <- c("Pooled","FE","IWE","RWE")
write.table(res.m2, file="tableA3_panelB.txt", row.names=FALSE)

stargazer(m2,m2fe,dep.var.labels.include = FALSE,type="text",
          out="tableA3_panelB_columns1and2.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))


# estimate and save results for far-right support
m3.iwe <- EstimateIWE(y="far_right", treatment="Distance", group="f.state", 
                      controls=c("prop_jewish25","unemployment33","population25","nazishare33"),
                      data=data.evs, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m3.fe <- rbind(m3.iwe$fe.est, sqrt(m3.iwe$fe.var), m3.iwe$fe.est/sqrt(m3.iwe$fe.var))
m3.iwe <- rbind(m3.iwe$swe.est, sqrt(m3.iwe$swe.var), m3.iwe$swe.est/sqrt(m3.iwe$swe.var))
m3.base <- rbind(summary(m3)$coefficients[2,1],summary(m3)$coefficients[2,2],summary(m3)$coefficients[2,3])

m3.rwe <- EstimateRWE(y="far_right", treatment="Distance", group="f.state", 
                      controls=c("prop_jewish25","unemployment33","population25","nazishare33"),
                      data=data.evs, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m3.rwe <- rbind(m3.rwe$swe.est, sqrt(m3.rwe$swe.var), m3.rwe$swe.est/sqrt(m3.rwe$swe.var))

res.m3 <- cbind(m3.base, m3.fe, m3.iwe, m3.rwe)
colnames(res.m3) <- c("Pooled","FE","IWE","RWE")
write.table(res.m3, file="tableA3_panelC.txt", row.names=FALSE)

stargazer(m3,m3fe,dep.var.labels.include = FALSE,type="text",
          out="tableA3_panelC_columns1and2.txt", no.space=T, 
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))




######### TABLE A4: WEIMAR-ERA LANDER

# Outcome: Intolerance toward outgroups

m1b <- lm(intolerance~Distance,evs_weimar)
m1bfe <- lm(intolerance~Distance + factor(weimarprov),evs_weimar)

#With Pre-treatment variables
m1<-lm(intolerance~Distance+prop_jewish25+unemployment33+population25+nazishare33 ,evs_weimar)
m1fe<-lm(intolerance~Distance+prop_jewish25+unemployment33+population25+nazishare33+factor(weimarprov) ,evs_weimar)

#With Pre/Post-treatment variables
m1f<-lm(intolerance~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          #Post-treatment
          lr+
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west,evs_weimar)
#With Pre/Post-treatment variables
m1ffe<-lm(intolerance~Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+
            factor(weimarprov)+
            #Post-treatment
            lr+
            immigrants07+
            unemployment07+
            unemp+
            educ+
            female+
            age+
            urban_scale+
            west,evs_weimar)

#G-estimator 1st stage
m1s<-lm(I(intolerance - 
            coef(m1f)['immigrants07']*immigrants07 -
            coef(m1f)['lr']*lr -
            coef(m1f)['unemp']*unemp - 
            coef(m1f)['unemployment07']*unemployment07 - 
            coef(m1f)['educ']*educ -
            coef(m1f)['urban_scale']*urban_scale) ~ 
          Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33, evs_weimar)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=6)
for(b in 1:boots){
  d.star <- evs_weimar[sample(1:nrow(evs_weimar), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(intolerance~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(intolerance - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33,
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:6]
}
#Extracting bootstrapped SEs
SEs1 <- apply(fl.boots,2,sd)


#G-estimator 1st stage
m1sfe<-lm(I(intolerance - 
              coef(m1ffe)['immigrants07']*immigrants07 -
              coef(m1ffe)['lr']*lr -
              coef(m1ffe)['unemp']*unemp - 
              coef(m1ffe)['unemployment07']*unemployment07 - 
              coef(m1ffe)['educ']*educ -
              coef(m1ffe)['urban_scale']*urban_scale) ~ 
            Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+factor(weimarprov), evs_weimar)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs_weimar[sample(1:nrow(evs_weimar), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(intolerance~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     factor(weimarprov)+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(intolerance - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33+
                      factor(weimarprov),
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs1fe <- apply(fl.boots,2,sd)


stargazer(m1b,m1bfe,m1,m1fe,m1s,m1sfe,se=list(NULL,NULL,NULL,NULL,SEs1,SEs1fe),dep.var.labels.include = FALSE,type="text",
          out="tableA4_panelA.txt", no.space=T, 
          keep = c("Distance","prop_jewish25","unemployment33","population25","nazishare33"),
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Land Fixed Effects","No","Yes","No","Yes","No","Yes"),
                         c("Post-Treatment Controls","No","No","No","No","Yes","Yes"),
                         c("Method","OLS","OLS","OLS","OLS","G-est","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))




# Outcome: Immigrant Resentment
m2b <- lm(resentment~Distance,evs_weimar)
m2bfe <- lm(resentment~Distance + factor(weimarprov),evs_weimar)

#With Pre-treatment variables
m2<-lm(resentment~Distance+prop_jewish25+unemployment33+population25+nazishare33 ,evs_weimar)
m2fe<-lm(resentment~Distance+prop_jewish25+unemployment33+population25+nazishare33+factor(weimarprov) ,evs_weimar)

#With Pre/Post-treatment variables
m2f<-lm(resentment~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          #Post-treatment
          lr+
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west,evs_weimar)

#With Pre/Post-treatment variables
m2ffe<-lm(resentment~Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+
            #Post-treatment
            lr+
            immigrants07+
            unemployment07+
            unemp+
            educ+
            female+
            age+
            urban_scale+
            west+factor(weimarprov) ,evs_weimar)


#G-estimator 1st stage
m2s<-lm(I(resentment - 
            coef(m2f)['immigrants07']*immigrants07 -
            coef(m2f)['lr']*lr -
            coef(m2f)['unemp']*unemp - 
            coef(m2f)['unemployment07']*unemployment07 - 
            coef(m2f)['educ']*educ -
            coef(m2f)['urban_scale']*urban_scale) ~ 
          Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33, evs_weimar)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs_weimar[sample(1:nrow(evs_weimar), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(resentment~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(resentment - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33,
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs2 <- apply(fl.boots,2,sd)


#G-estimator 1st stage
m2sfe<-lm(I(resentment - 
              coef(m2ffe)['immigrants07']*immigrants07 -
              coef(m2ffe)['lr']*lr -
              coef(m2ffe)['unemp']*unemp - 
              coef(m2ffe)['unemployment07']*unemployment07 - 
              coef(m2ffe)['educ']*educ -
              coef(m2ffe)['urban_scale']*urban_scale) ~ 
            Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+factor(weimarprov), evs_weimar)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs_weimar[sample(1:nrow(evs_weimar), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(resentment~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     factor(weimarprov)+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(resentment - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33+
                      factor(weimarprov),
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs2fe <- apply(fl.boots,2,sd)


stargazer(m2b,m2bfe,m2,m2fe,m2s,m2sfe,se=list(NULL,NULL,NULL,NULL,SEs2,SEs2fe),dep.var.labels.include = FALSE,type="text",
          out="tableA4_panelB.txt", no.space=T, 
          keep = c("Distance","prop_jewish25","unemployment33","population25","nazishare33"),
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Land Fixed Effects","No","Yes","No","Yes","No","Yes"),
                         c("Post-Treatment Controls","No","No","No","No","Yes","Yes"),
                         c("Method","OLS","OLS","OLS","OLS","G-est","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))




# Outcome: Support for Extreme Right-Wing Parties
m3b <- lm(far_right~Distance,evs_weimar)
m3bfe <- lm(far_right~Distance + factor(weimarprov),evs_weimar)

#With Pre-treatment variables
m3<-lm(far_right~Distance+prop_jewish25+unemployment33+population25+nazishare33 ,evs_weimar)
m3fe<-lm(far_right~Distance+prop_jewish25+unemployment33+population25+nazishare33+factor(weimarprov) ,evs_weimar)

#With Pre/Post-treatment variables
m3f<-lm(far_right~Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33+
          #Post-treatment
          lr+ 
          immigrants07+
          unemployment07+
          unemp+
          educ+
          female+
          age+
          urban_scale+
          west ,evs_weimar)

m3ffe<-lm(far_right~Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+
            #Post-treatment
            lr+ 
            immigrants07+
            unemployment07+
            unemp+
            educ+
            female+
            age+
            urban_scale+
            west+factor(weimarprov) ,evs_weimar)



#G-estimator 1st stage
m3s<-lm(I(far_right - 
            coef(m3f)['immigrants07']*immigrants07 -
            coef(m3f)['lr']*lr -
            coef(m3f)['unemp']*unemp - 
            coef(m3f)['unemployment07']*unemployment07 - 
            coef(m3f)['educ']*educ -
            coef(m3f)['urban_scale']*urban_scale) ~ 
          Distance+
          prop_jewish25+
          unemployment33+
          population25+
          nazishare33, evs_weimar)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs_weimar[sample(1:nrow(evs_weimar), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(far_right~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(far_right - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33,
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs3 <- apply(fl.boots,2,sd)


#G-estimator 1st stage
m3sfe<-lm(I(far_right - 
              coef(m3ffe)['immigrants07']*immigrants07 -
              coef(m3ffe)['lr']*lr -
              coef(m3ffe)['unemp']*unemp - 
              coef(m3ffe)['unemployment07']*unemployment07 - 
              coef(m3ffe)['educ']*educ -
              coef(m3ffe)['urban_scale']*urban_scale) ~ 
            Distance+
            prop_jewish25+
            unemployment33+
            population25+
            nazishare33+factor(weimarprov), evs_weimar)

#Full G-estimator
boots <- 1000
set.seed(543)
fl.boots <- matrix(NA, nrow=boots, ncol=21)
for(b in 1:boots){
  d.star <- evs_weimar[sample(1:nrow(evs_weimar), replace=TRUE),]
  #G-estimator 1st stage
  boot.first <- lm(far_right~
                     Distance+
                     prop_jewish25+
                     unemployment33+
                     population25+
                     nazishare33+
                     factor(weimarprov)+
                     #Post-treatment
                     lr+ 
                     immigrants07+
                     unemployment07+
                     unemp+
                     educ+
                     female+
                     age+
                     urban_scale+
                     west,
                   d.star)
  #G-estimator 2nd stage
  boot.direct <- lm(I(far_right - 
                        coef(boot.first)['immigrants07']*immigrants07 -
                        coef(boot.first)['lr']*lr -
                        coef(boot.first)['unemp']*unemp - 
                        coef(boot.first)['unemployment07']*unemployment07 - 
                        coef(boot.first)['educ']*educ -
                        coef(boot.first)['urban_scale']*urban_scale) ~ 
                      Distance+
                      prop_jewish25+
                      unemployment33+
                      population25+
                      nazishare33+
                      factor(weimarprov),
                    d.star)
  fl.boots[b,] <- coef(boot.direct)[1:21]
}
#Extracting bootstrapped SEs
SEs3fe <- apply(fl.boots,2,sd)



stargazer(m3b,m3bfe,m3,m3fe,m3s,m3sfe,se=list(NULL,NULL,NULL,NULL,SEs3,SEs3fe),dep.var.labels.include = FALSE,type="text",
          out="tableA4_panelC.txt", no.space=T, 
          keep = c("Distance","prop_jewish25","unemployment33","population25","nazishare33"),
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Land Fixed Effects","No","Yes","No","Yes","No","Yes"),
                         c("Post-Treatment Controls","No","No","No","No","Yes","Yes"),
                         c("Method","OLS","OLS","OLS","OLS","G-est","G-est")), 
          covariate.labels = c('Distance to camp', '\\% Jews (1925)', '\\% Unemployed (1933)',
                               'Population (1925)', 'Nazi party share (1933)'))




############### TABLE 2 and TABLE A5

##  HAUSMAN TESTS

form <- AfDshare ~ distance2 + prop_juden + unemp33 + pop25 + vshare33
wi<- plm(form, data = elect_data, model = "within", index="NAME_1")
re <- plm(form, data = elect_data, model = "random", index="NAME_1")
pooled <- plm(form, data = elect_data, model = "pooling", index="NAME_1")
re.v.pooled <- round(phtest(re, pooled)$p.value, digits=3)
fe.v.pooled <- round(phtest(wi, pooled)$p.value, digits=3)
fe.v.re <- round(phtest(wi, re)$p.value, digits=3)

form <- AfDNPDshare ~ distance2 + prop_juden + unemp33 + pop25 + vshare33
wi2<- plm(form, data = elect_data, model = "within", index="NAME_1")
re2 <- plm(form, data = elect_data, model = "random", index="NAME_1")
pooled2 <- plm(form, data = elect_data, model = "pooling", index="NAME_1")
re.v.pooled2 <- round(phtest(re2, pooled2)$p.value, digits=3)
fe.v.pooled2 <- round(phtest(wi2, pooled2)$p.value, digits=3)
fe.v.re2 <- round(phtest(wi2, re2)$p.value, digits=3)


stargazer(pooled,wi, 
          out="table2_panelA.txt", no.space=T, keep="distance2",
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","Pooled","FE"),
                         c("FE v Pooled",fe.v.pooled,"","")))

stargazer(pooled2,wi2, 
          out="table2_panelB.txt", no.space=T, keep="distance2",
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","Pooled","FE"),
                         c("FE v Pooled",fe.v.pooled2,"","")))


mod1e <- EstimateIWE(y="AfDshare", treatment="distance2", group="NAME_1", 
                     controls=c("prop_juden","unemp33","pop25","vshare33"),
                     data=elect_data_bfe, subset=NULL, cluster.var=NULL,is.robust=TRUE)
mod1f <- EstimateRWE(y="AfDshare", treatment="distance2", group="NAME_1", 
                     controls=c("prop_juden","unemp33","pop25","vshare33"),
                     data=elect_data_bfe, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m1.iwe <- rbind(mod1e$swe.est, sqrt(mod1e$swe.var), mod1e$swe.est/sqrt(mod1e$swe.var))
m1.rwe <- rbind(mod1f$swe.est, sqrt(mod1f$swe.var), mod1f$swe.est/sqrt(mod1f$swe.var))

res.m1 <- cbind(m1.iwe, m1.rwe)
colnames(res.m1) <- c("IWE","RWE")

mod2e <- EstimateIWE(y="AfDNPDshare", treatment="distance2", group="NAME_1", 
                     controls=c("prop_juden","unemp33","pop25","vshare33"),
                     data=elect_data_bfe, subset=NULL, cluster.var=NULL,is.robust=TRUE)
mod2f <- EstimateRWE(y="AfDNPDshare", treatment="distance2", group="NAME_1", 
                     controls=c("prop_juden","unemp33","pop25","vshare33"),
                     data=elect_data_bfe, subset=NULL, cluster.var=NULL,is.robust=TRUE)
m2.iwe <- rbind(mod2e$swe.est, sqrt(mod2e$swe.var), mod2e$swe.est/sqrt(mod2e$swe.var))
m2.rwe <- rbind(mod2f$swe.est, sqrt(mod2f$swe.var), mod2f$swe.est/sqrt(mod2f$swe.var))

res.m2 <- cbind(m2.iwe, m2.rwe)
colnames(res.m2) <- c("IWE","RWE")

write.table(round(cbind(res.m1,res.m2),3), file="table2_columns3and4.txt", row.names=FALSE)


stargazer(pooled,wi, 
          out="tableA5_panelA.txt", no.space=T,
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","Pooled","FE"),
                         c("FE v Pooled",fe.v.pooled,"","")))

stargazer(pooled2,wi2, 
          out="tableA5_panelB.txt", no.space=T,
          star.cutoffs = c(0.05, 0.01, NA), keep.stat = c("n","adj.rsq"),
          add.lines=list(c("Method","Pooled","FE"),
                         c("FE v Pooled",fe.v.pooled2,"","")))


