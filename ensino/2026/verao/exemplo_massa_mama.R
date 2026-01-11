rm(list = ls())
library(glmnet)
set.seed(2025)
# https://runzelipsu.github.io/DataScience/dataset/ch5/mamm_des.txt

# Baixar dados
data_mama <- read.table("https://runzelipsu.github.io/DataScience/dataset/ch5/mamm_d.dat", 
                        header=FALSE, skip=0, sep = ',')

dim(data_mama)

colnames(data_mama) <- c('birads', 'idade', 'forma', 'margem', 'densidade', 
                         'severidade')
head(data_mama)
data_mama$forma <- as.factor(data_mama$forma)
data_mama$margem <- as.factor(data_mama$margem)
table(data_mama$densidade)

n <- dim(data_mama)[1]

# variável resposta
y <- data_mama$severidade

# matriz de covariáveis
p <- 10
X <- matrix(0, nrow = n, ncol = p)
X[,1] <- data_mama$birads
X[,2] <- data_mama$idade
X[,3] <- data_mama$densidade
X[,4] <- (data_mama$forma==1)
X[,5] <- (data_mama$forma==2)
X[,6] <- (data_mama$forma==3)
X[,7] <- (data_mama$margem==1)
X[,8] <- (data_mama$margem==2)
X[,9] <- (data_mama$margem==3)
X[,10] <- (data_mama$margem==4)


colnames(X) <- c('birads', 'idade', 'densidade', 'circular', 'oval', 'lobular',
                 'circunscrito', 'microlobulado', 'obscuro', 'maldef')





###################################
# Ajuste com MLG usual

mod0 <- glm(y~1+X)
summary(mod0)

k <- 2
exp(coef(mod0)[k+1])


###################################
# ajuste do lasso
lasso_fit <- glmnet::glmnet(X, y, family = 'binomial', intercept = TRUE,
                            alpha = 1)

# png(file="mama_lasso_path.png", 
#     width=600, height=500, res = 100)
plot(lasso_fit, xvar = 'lambda', sign.lambda = 1, 
     lwd=2, ylab='Coeficientes')
# dev.off()

# validação cruzada para escolher lambda
cv_lasso_fit <- glmnet::cv.glmnet(X, y, family = 'binomial', intercept = TRUE,
                                  alpha = 1, type.measure = 'deviance', nfolds = 10) 

# png(file="mama_lasso_deviance_cv.png", 
#     width=600, height=500, res = 100)
plot(cv_lasso_fit, sign.lambda = 1)
# dev.off()

# coeficientes estimados usando o lambda 'ótimo'
cv_lasso_fit$lambda.min
id_labda_min <- which(lasso_fit$lambda == cv_lasso_fit$lambda.min)
beta_hat_lasso <- coef(lasso_fit, s=lasso_fit$lambda[id_labda_min])

# valores ajustados
fitted_eta_values <- beta_hat_lasso[1] + X%*%beta_hat_lasso[-1]
fitted_values <- exp(fitted_eta_values)/(1 + exp(fitted_eta_values))

# deviance
-2*sum(y*log(fitted_values) + (1-y)*log(1 - fitted_values))
-2*sum(y*log(mean(y)) + (1-y)*log(1 - mean(y)))
# fração da deviance explicada nos dados de treino
dev_lambda <- -2*sum(y*log(fitted_values) + (1-y)*log(1 - fitted_values))
dev_null <- -2*sum(y*log(mean(y)) + (1-y)*log(1 - mean(y)))
1 - dev_lambda/dev_null

# razão de chances usual para a covariável idade
k <- 1
exp(beta_hat_lasso[k+1])

exp(beta_hat_lasso[k+1]/mean(X[,k]^2))


# Matriz com estimativas e razões de chance, para o MLG-lasso e para o MLG usual
cbind(rownames(beta_hat_lasso), 
      round(as.vector(beta_hat_lasso), 2),
      round(as.vector(exp(beta_hat_lasso)), 2),
      round(mod0$coefficients, 2),
      round(exp(mod0$coefficients), 2)
)
