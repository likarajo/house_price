if(!require(MASS)){install.packages("MASS")}
library(MASS)

if(!require(ISLR)){install.packages("ISLR")}
library(ISLR)

if(!require(tidyverse)){install.packages("tidyverse")}
library(tidyverse)

if(!require(ggplot2)){install.packages("ggplot2")}
library(ggplot2)

if(!require(gridExtra)){install.packages("gridExtra")}
library(gridExtra)

# Read the data

# can be downloaded from http://www.utdallas.edu/~rxc170010/housing.csv)
housing <- read.csv('housing.csv')

# Explore dataset

nrow(housing)
ncol(housing)
colnames(housing)
str(housing)
head(housing)
summary(housing)

# Explore variables

h1 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = latitude), bins=30, color="black")
h2 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = longitude), bins=30, color="black")
h3 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = housing_median_age), bins=30, color="black")
h4 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = total_rooms), bins=30, color="black")
h5 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = total_bedrooms), bins=30, color="black")
h6 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = population), bins=30, color="black")
h7 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = households), bins=30, color="black")
h8 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = median_income), bins=30, color="black")
h9 <- ggplot(data = housing) + geom_histogram(mapping = aes(x = median_house_value), bins=30, color="black")
grid.arrange(h1, h2, h3, h4, h4, h6, h7, h8, h9, nrow = 3)

# Handle variables

# Fill NA's in total_bedrooms with the median (since mean is affected by outliers)
housing$total_bedrooms[is.na(housing$total_bedrooms)] <- median(housing$total_bedrooms, na.rm=TRUE)

# Replace total_bedrooms and total_rooms with avg_bedrooms and avg_bedrooms
housing$avg_bedrooms <- housing$total_bedrooms/housing$households
housing$avg_rooms <- housing$total_rooms/housing$households
housing <- housing[, !(names(housing) %in% c('total_bedrooms', 'total_rooms'))]

# Split the ocean_proximity into seperate boolean category columns
categories <- unique(housing$ocean_proximity)
for(c in categories){
  housing[,c] <- rep(0, times=nrow(housing))
}
for (i in 1:nrow(housing)){
  c <- as.character(housing$ocean_proximity[i])
  housing[, c][i] <- 1
}
housing$ocean_proximity <- NULL

# Check variables
colnames(housing)
head(housing)
str(housing)

# Check relationship of the variables with median_house_value

s1 <- ggplot(data = housing) + geom_point(mapping = aes(x = longitude, y = median_house_value))
s2 <- ggplot(data = housing) + geom_point(mapping = aes(x = latitude, y = median_house_value))
s3 <- ggplot(data = housing) + geom_point(mapping = aes(x = housing_median_age, y = median_house_value))
s4 <- ggplot(data = housing) + geom_point(mapping = aes(x = population, y = median_house_value))
s5 <- ggplot(data = housing) + geom_point(mapping = aes(x = households, y = median_house_value))
s6 <- ggplot(data = housing) + geom_point(mapping = aes(x = median_income, y = median_house_value))
s7 <- ggplot(data = housing) + geom_point(mapping = aes(x = median_house_value, y = median_house_value))
s8 <- ggplot(data = housing) + geom_point(mapping = aes(x = avg_bedrooms, y = median_house_value))
s9 <- ggplot(data = housing) + geom_point(mapping = aes(x = avg_rooms, y = median_house_value))
s10 <- ggplot(data = housing) + geom_point(mapping = aes(x = `NEAR BAY`, y = median_house_value))
s11 <- ggplot(data = housing) + geom_point(mapping = aes(x = INLAND, y = median_house_value))
s12 <- ggplot(data = housing) + geom_point(mapping = aes(x = `NEAR OCEAN`, y = median_house_value))
s13 <- ggplot(data = housing) + geom_point(mapping = aes(x = ISLAND, y = median_house_value))
s14 <- ggplot(data = housing) + geom_point(mapping = aes(x = `<1H OCEAN`, y = median_house_value))
grid.arrange(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, nrow = 5)

# Check for outliers 

b1 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = longitude))
b2 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = latitude))
b3 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = housing_median_age))
b4 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = population))
b5 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = households))
b6 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = median_income))
b7 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = median_house_value))
b8 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = avg_bedrooms))
b9 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = avg_rooms))
b10 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = `NEAR BAY`))
b11 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = INLAND))
b12 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = `NEAR OCEAN`))
b13 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = ISLAND))
b14 <- ggplot(data = housing) + geom_boxplot(mapping = aes(x = `<1H OCEAN`))
grid.arrange(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, nrow = 5)

# Check correlation of the variables

install.packages("corrplot")
require(corrplot)
par(mfrow=c(1,1))
corrplot(cor(housing), method="square")
corMat <- as.data.frame(corrplot(cor(housing), method="number"))

# Find the variable with correlation >60% with median_house_value
row.names(corMat)[abs(corMat$median_house_value) > 0.60]

# Create model to predict the median_house_value using the variable(s) having >60% correlation
lm.fit <- lm(median_house_value~median_income, data=housing)

lm.fit
summary(lm.fit)
names(lm.fit)
confint(lm.fit)

plot(housing$median_income, housing$median_house_value, xlab = "median_income", ylab="median_house_value", col="blue")
abline(lm.fit, lwd=3, col="red")

# Randomize and Split data to training (75%) and test (25%) sets
sample.size <- floor(0.75 * nrow(housing))
set.seed(111)
idx <- sample(seq_len(nrow(housing)), size=sample.size)
trainData <- housing[idx,]
testData <- housing[-idx,]

# Train model with training data
lm2.fit <- lm(median_house_value~median_income, data=trainData)

lm2.fit
summary(lm2.fit)
names(lm2.fit)
confint(lm2.fit)

# Predict using test data
preds <- predict(lm2.fit, testData)

# Plot test data and predictions
plot(testData$median_income, testData$median_house_value, xlab = "median_income (test)", ylab="median_house_value (test)", col="green")
abline(lm2.fit, lwd=3, col="blue")

par(mfrow=c(2,1))
hist(testData$median_house_value, ylim = c(0,2000), col = rgb(0,0.5,0,1))

hist(testData$median_house_value, ylim = c(0,2000), col = rgb(0,0.5,0,1))
hist(preds, col = rgb(0,0.5,0,0.5), add=T)

# Find error
install.packages("Metrics")
library(Metrics)
mse <- mse(housing$median_house_value, preds)
rmse <- sqrt(mse)
rmse
