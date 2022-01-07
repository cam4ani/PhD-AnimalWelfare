library("MCMCglmm")
library("asreml")
data("BTdata")      # load data
data("BTped")       # load pedigree

nitt <- 60000
burnin <- 10000
thin <- 25


usP<-list(V = diag(2)/3, nu = 2)
idhP<-list(V = diag(2)*2/3, nu = 2)

# marginal prior: nu* = 1 ,  V* diag(2)*(2/3)
# so idh structures should have nu = 2 V = diag(2)*(2/3)

prior1 <- list(R = usP, G = list(G1 = usP, G2 = usP))                 

m1 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + us(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior1,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m1b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + us(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior1,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

prior2 <- list(R = usP, G = list(G1 = idhP, G2 = usP))                 

m2 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ idh(trait):animal + us(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior2,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m2b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ idh(trait):animal + us(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior2,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)


prior3 <- list(R = usP, G = list(G1 = idhP, G2 = idhP))  

m3 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ idh(trait):animal + idh(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior3,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m3b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ idh(trait):animal + idh(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior3,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

prior4 <- list(R = idhP, G = list(G1 = idhP, G2 = idhP))  

m4 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ idh(trait):animal + idh(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior4,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m4b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ idh(trait):animal + idh(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior4,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

prior5 <- list(R = usP, G = list(G1 = usP, G2 = idhP))  

m5 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + idh(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior5,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)


m5b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + idh(trait):fosternest, 
      rcov = ~ us(trait):units, 
      prior = prior5,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

prior6 <- list(R = idhP, G = list(G1 = usP, G2 = idhP))  

m6 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + idh(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior6,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m6b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + idh(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior6,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)


prior7 <- list(R = idhP, G = list(G1 = idhP, G2 = usP))  

m7 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + idh(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior7,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m7b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + idh(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior7,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

prior8 <- list(R = idhP, G = list(G1 = usP, G2 = usP))  

m8 <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + us(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior8,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

m8b <- MCMCglmm(
      fixed = cbind(tarsus, back) ~ trait:sex + trait:hatchdate - 1, 
      random = ~ us(trait):animal + us(trait):fosternest, 
      rcov = ~ idh(trait):units, 
      prior = prior8,
      family = c("gaussian", "gaussian"),
      nitt = nitt, burnin = burnin, thin = thin,
      data = BTdata, pedigree = BTped)

