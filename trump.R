
library(dplyr)
require(magrittr)
library(tm)
library(ggplot2)
library(stringr)
library(NLP)
library(tidytext)
library(quanteda)
library(stm)
library(plyr)
library(syuzhet)
require(quanteda)

#load .csv file with news articles
precorpus<- read.csv("Speeches.csv", header=TRUE, stringsAsFactors=FALSE)
corpus <- corpus(precorpus$Text,docnames=precorpus$Date)
summary(corpus)
#passing Full Text to variable news_2015
news_2015<-precorpus$Text




summary(precorpus)

browseVignettes(package = "tidytext")

precorpus['text'] <- precorpus$Text


nrc <- get_sentiments("nrc")
#tokenize at word level 
tidy_news <- precorpus %>%
  unnest_tokens(word, text)
tidy_news$Text <- NULL
#remove stopworks 
data("stop_words")
cleaned_trump <- tidy_news %>%
  anti_join(stop_words)

#add sentiments
nrc <- get_sentiments("nrc")
bing <- get_sentiments("bing")
trump_sentiment <- get_sentiment(cleaned_trump$word, method="bing")
cleaned_trump$score <- get_sentiment(cleaned_trump$word, method="bing")
summary(trump_sentiment)
cleaned_trump %>%
  join(nrc) %>%
  count(word, sort=TRUE)






newnew <- merge(cleaned_trump, bing, by ="word")
newplussent <- merge(newnew, nrc, by="word")

library(plyr)
newnew$sentiment <- as.factor(newnew$sentiment)
y <- count(newplussent, 'sentiment.y')

y
table(newnew$sentiment)
table(newplussent$sentiment.x)
x <- table(newplussent$sentiment.y)
x <- as.data.frame(x)

write.csv(newplussent, file="newnew.csv")
hist(sum(newplussent$sentiment.y))
hist(newplussent$sentiment.x)
plot(x)


bing_word_counts1 <- cleaned_trump %>%
  inner_join(bing) %>%
  count(word, sentiment, sort = TRUE) %>%
  ungroup()

bing_word_counts

ggplot(bing_word_counts, aes(fill = sentiment)) 
  
library(tidyr)
library(ggplot2)
bing_word_counts1 %>%
  filter(n > 150) %>%
  mutate(n = ifelse(sentiment == "negative", -n, n)) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n, fill = sentiment)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ylab("Contribution to sentiment")

library(reshape2)
library(wordcloud)
help(comparison.cloud)
cleaned_trump %>%
  inner_join(bing) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("#F8766D", "#00BFC4"),
                   max.words = 100)

#Cleaning corpus
stop_words <- stopwords("SMART")
## additional junk words showing up in the data
stop_words <- c(stop_words, "will", "one", "also")
stop_words <- tolower(stop_words)


news_2015 <- gsub("'", "", news_2015) # remove apostrophes
news_2015 <- gsub("[[:punct:]]", " ", news_2015)  # replace punctuation with space
news_2015 <- gsub("[[:cntrl:]]", " ", news_2015)  # replace control characters with space
news_2015 <- gsub("^[[:space:]]+", "", news_2015) # remove whitespace at beginning of documents
news_2015 <- gsub("[[:space:]]+$", "", news_2015) # remove whitespace at end of documents
news_2015 <- gsub("[^a-zA-Z -]", " ", news_2015) # allows only letters
news_2015 <- tolower(news_2015)  # force to lowercase

## get rid of blank docs
news_2015 <- news_2015[news_2015 != ""]

# tokenize on space and output as a list:
doc.list <- strsplit(news_2015, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# remove terms that are stop words or occur fewer than 5 times:
del <- names(term.table) %in% stop_words | term.table < 5
term.table <- term.table[!del]
term.table <- term.table[names(term.table) != ""]
vocab <- names(term.table)

# now put the documents into the format required by the lda package:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

#############
# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (1)
W <- length(vocab)  # number of terms in the vocab (1741)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (56196)
term.frequency <- as.integer(term.table) 

# MCMC and model tuning parameters:
K <- 10
G <- 3000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
## display runtime
t2 - t1  

theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

news_for_LDA <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)

library(LDAvis)
library(servr)

# create the JSON object to feed the visualization:
json <- createJSON(phi = news_for_LDA$phi, 
                   theta = news_for_LDA$theta, 
                   doc.length = news_for_LDA$doc.length, 
                   vocab = news_for_LDA$vocab, 
                   term.frequency = news_for_LDA$term.frequency)

serVis(json, out.dir = 'vis', open.browser = TRUE)

