library(ggplot2)
library(readr)
library(dplyr)

survey<-read.csv(file="../data/survey_means_weighted.csv",header=TRUE,row.names=1)
survey <- subset(survey, select=c("gender_mean","class_mean"))
df<-read.csv(file="../data/GoogleNews_Fake.csv", header=TRUE,row.names=1, sep=",")
#df<-read.csv(file="/DIRECTORY/US_Ngrams_2000_12.csv", header=TRUE,row.names=1, sep=",")

## New
#survey<- readr::read_csv(file="../data/survey_means_weighted.csv", show_col_types=FALSE)
##df<- readr::read_csv(file="../data/GoogleNews_Embedding.csv", show_col_types=FALSE)
#df <- readr::read_csv(file="../data/GoogleNews_Fake.csv", show_col_types=FALSE)
#num_numeric_cols <- ncol(df) - 1
#names(df) <- c("token", paste0("x",1:num_numeric_cols))
##df<-read.csv(file="/DIRECTORY/US_Ngrams_2000_12.csv", header=TRUE,row.names=1, sep=",")


#####DEFINE FUNCTIONS##########
#Calculate norm of vector#
norm_vec <- function(x) sqrt(sum(x^2))

#Dot product#
dot <- function(x,y) (sum(x*y))

#Cosine Similarity#
cos <- function(x,y) dot(x,y)/norm_vec(x)/norm_vec(y)

#Normalize vector#
nrm <- function(x) x/norm_vec(x)

#Calculate semantic dimension from antonym pair#
dimension<-function(x,y) nrm(nrm(x)-nrm(y))

###IMPORT LISTS OF TERMS TO PROJECT AND ANTONYM PAIRS#####
ant_pairs_aff <- read.csv("../data/word_pairs/affluence_pairs.csv",header=FALSE, stringsAsFactor=F)
ant_pairs_gen <- read.csv("../data/word_pairs/gender_pairs.csv",header=FALSE, stringsAsFactor=F)
ant_pairs_gen[11,] <- c("king","queen")
#ant_pairs_race <- read.csv("../data/word_pairs/race_pairs.csv",header=FALSE, stringsAsFactor=F)


###STORE EMBEDDING AS MATRIX, NORMALIZE WORD VECTORS###
cdfm<-as.matrix(data.frame(df))
cdfmn<-t(apply(cdfm,1,nrm))


word_dims<-matrix(NA,nrow(ant_pairs_gen),2)


###SETUP "make_dim" FUNCTION, INPUT EMBEDDING AND ANTONYM PAIR LIST#######
###OUTPUT AVERAGE SEMANTIC DIMENSION###

normalize <- function(orig_emb) {
  return(orig_emb / sum(orig_emb))
}

# Original...

make_dim<-function(embedding,pairs){
  word_dims<-data.frame(matrix(NA,nrow(pairs),2))
  for (j in 1:nrow(pairs)){
    rp_word1<-pairs[j,1]
    rp_word2<-pairs[j,2]
    tryCatch(word_dims[j,]<-dimension(embedding[rp_word1,],embedding[rp_word2,]),error=function(e){})
  }
  dim_ave<-colMeans(word_dims, na.rm = TRUE)
  dim_ave_n<-nrm(dim_ave)
  return(dim_ave_n)
}

make_dim_new <- function(embedding, pairs) {
  # Empty matrix with one row per pair
  dim_matrix <- matrix(NA, nrow(pairs), 300)
  #word_dims <- data.frame(matrix(NA,nrow(pairs),300))
  for (i in nrow(pairs)) {
    word1 <- pairs[i,1]
    word2 <- pairs[i,2]
    w1_row <- embedding %>% filter(token == word1)
    w1_emb <- w1_row %>% select(-token)
    w1_emb <- normalize(w1_emb)
    w2_row <- embedding %>% filter(token == word2)
    w2_emb <- w2_row %>% select(-token)
    w2_emb <- normalize(w2_emb)
    pair_dim <- normalize(w1_emb - w2_emb)
    dim_list[[i]] = pair_dim
  }
  return(dim_list)
}


#####CONSTRUCT AFFLUENCE, GENDER, AND RACE DIMENSIONS######
aff_dim<-make_dim(df,ant_pairs_aff)
gender_dim<-make_dim(df,ant_pairs_gen)
#race_dim<-make_dim(df,ant_pairs_race)


####ANGLES BETWEEN DIMENSIONS#######
cos(aff_dim,gender_dim)
#cos(aff_dim,race_dim)
#cos(gender_dim,race_dim)


####CALCULATE PROJECTIONS BY MATRIX MULTIPLICATION####
###(Equivalent to cosine similarity because vectors are normalized)###
aff_proj<-cdfmn%*%aff_dim
gender_proj<-cdfmn%*%gender_dim
#race_proj<-cdfmn%*%race_dim

projections_df<-cbind(aff_proj, gender_proj)
colnames(projections_df)<-c("aff_proj","gender_proj")


####MERGE WITH SURVEY AND CALCULATE CORRELATION####
projections_sub<-subset(projections_df, rownames(projections_df) %in% rownames(survey))
colnames(projections_sub)<-c("aff_proj","gender_proj")
survey_proj<-merge(survey,projections_sub,by=0)


# ?? different names?
#cor(survey_proj$survey_class,survey_proj$aff_proj)
cor(survey_proj$class_mean, survey_proj$aff_proj)
#cor(survey_proj$survey_gender,survey_proj$gender_proj)
cor(survey_proj$gender_mean, survey_proj$gender_proj)
##cor(survey_proj$survey_race,survey_proj$race_proj)

########################################################################


###CREATE VISUALIZATION###
#wlist=c("camping","baseball","boxing","volleyball","softball","golf","tennis","soccer","basketball","hockey")
wlist = c("tennis","rugby","man","woman","rich","poor")
Visualization<-ggplot(data=data.frame(projections_df[wlist,]),aes(x=gender_proj,y=aff_proj,label=wlist)) + geom_text()
Visualization+ theme_bw() +ylim(-1,1) +xlim(-1,1)

########################################################################











# ### New
#
# # Normalize embedding df
# row_sums <- rowSums(select(df, -token))
# df_norm <- df %>% select(-token) %>% mutate_all(~ ./row_sums)
# df <- dplyr::bind_cols(df$token, df_norm)
# names(df)[1] <- "token"
#
# word_dims <- make_dim(df, ant_pairs_aff)
# # And normalize the dim_list
# #dim_mean <- colMeans(word_dims, na.rm = TRUE)
# #dim_mean_norm <- nrm(dim_ave)
#
# #####CONSTRUCT AFFLUENCE, GENDER, AND RACE DIMENSIONS######
# aff_dim<-make_dim(df,ant_pairs_aff)
# gender_dim<-make_dim(df,ant_pairs_gen)
# race_dim<-make_dim(df,ant_pairs_race)
