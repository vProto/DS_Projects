#!/usr/bin/env python
# coding: utf-8

# # Recommendations with IBM
# 
# In this notebook, you will be putting your recommendation skills to use on real data from the IBM Watson Studio platform. 
# 
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](Need to update this).  **Please save regularly.**
# 
# By following the table of contents, you will build out a number of different methods for making recommendations that can be used for different situations. 
# 
# 
# ## Table of Contents
# 
# I. [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
# II. [Rank Based Recommendations](#Rank)<br>
# III. [User-User Based Collaborative Filtering](#User-User)<br>
# IV. [Content Based Recommendations](#Content-Recs)<br>
# V. [Matrix Factorization](#Matrix-Fact)<br>
# VI. [Extras & Concluding](#conclusions)
# 
# At the end of the notebook, you will find directions for how to submit your work.  Let's get started by importing the necessary libraries and reading in the data.

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import plotly.graph_objects as go
import time
# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords',
               'averaged_perceptron_tagger'])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import re



df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()


# In[2]:


# Show df_content to get an idea of the data
df_content.head()


# ### <a class="anchor" id="Exploratory-Data-Analysis">Part I : Exploratory Data Analysis</a>
# 
# Use the dictionary and cells below to provide some insight into the descriptive statistics of the data.
# 
# `1.` What is the distribution of how many articles a user interacts with in the dataset?  Provide visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.  

# In[3]:


df.info() 


# In[4]:


x = df.groupby(["email"])["article_id"].count()

fig = go.Figure()
fig.add_trace(go.Histogram(x=x))


fig.update_layout(
    title_text='Distribution of how many articles a user interacts', # title of plot
    xaxis_title_text='Articles', # xaxis label
    yaxis_title_text='Count' # yaxis label

)

fig.show()


# In[5]:


# Fill in the median and maximum number of user_article interactios below

median_val =x.median() # 50% of individuals interact with ____ number of articles or fewer.
max_views_by_user =x.max() # The maximum number of user-article interactions by any 1 user is ______.


# `2.` Explore and remove duplicate articles from the **df_content** dataframe.  

# In[6]:


# Find and explore duplicate articles
df_content[df_content.duplicated(["article_id"])]


# In[7]:


# Remove any rows that have the same article_id - only keep the first
df_content = df_content.drop_duplicates(["article_id"],keep="first")


# `3.` Use the cells below to find:
# 
# **a.** The number of unique articles that have an interaction with a user.  
# **b.** The number of unique articles in the dataset (whether they have any interactions or not).<br>
# **c.** The number of unique users in the dataset. (excluding null values)<br> 
# **d.** The number of user-article interactions in the dataset.

# In[8]:


unique_articles = df.article_id.unique().shape[0]# The number of unique articles that have at least one interaction
total_articles =df_content.shape[0] # The number of unique articles on the IBM platform
unique_users =df.loc[~df.email.isna(),"email"].unique().shape[0] # The number of unique users
user_article_interactions =df.shape[0] # The number of user-article interactions


# `4.` Use the cells below to find the most viewed **article_id**, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was find using other information that all of these null values likely belonged to a single user.

# In[9]:


most_viewed_article_id = df.value_counts("article_id").idxmax().astype(str)# The most viewed article in the dataset as a string with one value following the decimal 
max_views = df.value_counts("article_id").max()# The most viewed article in the dataset was viewed how many times?


# In[10]:


## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()


# In[11]:


## If you stored all your results in the variable names above, 
## you shouldn't need to change anything in this cell

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)


# ### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>
# 
# Unlike in the earlier lessons, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article.  In these cases, the popularity of an article can really only be based on how often an article was interacted with.
# 
# `1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.

# In[12]:


def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # Your code here
    counts = df.groupby(["article_id"])[["title"]].count().reset_index().sort_values("title",ascending=False)
    counts = counts.head(n)
    top_articles = counts[["article_id"]].merge(df[["article_id","title"]].drop_duplicates(),on="article_id").title
    top_articles = list(top_articles)
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    
    '''
    # Your code here
    counts = df.groupby(["article_id"])["article_id"].count().sort_values(ascending=False)
    top_articles = list(counts.index.astype(str)[0:n])
    return top_articles # Return the top article ids


# In[13]:


print(get_top_articles(10))
print(get_top_article_ids(10))


# In[14]:


# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

# Test each of your three lists from above
t.sol_2_test(get_top_articles)


# ### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>
# 
# 
# `1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.  
# 
# * Each **user** should only appear in each **row** once.
# 
# 
# * Each **article** should only show up in one **column**.  
# 
# 
# * **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.  
# 
# 
# * **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**. 
# 
# Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.

# In[15]:


# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''

    user_article_df = df.drop_duplicates(subset=['article_id', 'user_id']).groupby(['user_id','article_id'])['title'].count().unstack(fill_value=0)
    
    return user_article_df # return the user_item matrix 

user_item = create_user_item_matrix(df)


# In[16]:


## Tests: You should just need to run this cell.  Don't change the code.
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("You have passed our quick tests!  Please proceed!")


# `2.` Complete the function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users. 
# 
# Use the tests to test your function.

# In[17]:


def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # compute similarity of each user to the provided user
    similarity = user_item.dot(user_item.loc[user_id])
    # sort by similarity
    similarity = similarity.sort_values(ascending=False)
    # create list of just the ids
    similar_ids = list(similarity.index)

    # remove the own user's id
    similar_ids.remove(user_id)
    
       
    return similar_ids # return a list of the users in order from most to least similar
        


# In[18]:


# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))
print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933)[:5]))
print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46)[:3]))


# `3.` Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.  Complete the functions below to return the articles you would recommend to each user. 

# In[19]:


def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    # Your code here
    temp = df[["article_id","title"]].copy()
    temp = temp.astype(str)
    temp = temp.loc[temp.article_id.isin(article_ids),["article_id","title"]].drop_duplicates()
    

    article_names = list(temp.title)
    
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # Your code here
    temp = user_item.loc[user_id]
    article_ids = list(temp[temp!=0].index.astype(str))
    article_names = get_article_names(article_ids)
    
    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # Your code here
    similar_users = find_similar_users(user_id)
    articles_read = get_user_articles(user_id)
 
    
    
    recs_left = m # indicator of recommendations left to make
    recs = [] # list that will store the recommendation ids
    i = 0 # counter

    # while we still have room for recommendations
    while recs_left>0:
        user = similar_users[i] # take the most similar user
        
        # articles that similar user has read
        candidate_reads_ids,_  = get_user_articles(user)
        
        # remove articles that the user has already seen
        candidate_reads_ids = np.setdiff1d(np.array(candidate_reads_ids), np.array(articles_read))
        
        # if the recommendations are more than the room we have
        if len(candidate_reads_ids)>recs_left:
           # add to the list
            recs.extend(candidate_reads_ids[:recs_left])
            recs_left = recs_left - len(candidate_reads_ids[:recs_left])
        
        else:
            # add to the list
            recs.append(candidate_reads_ids)
            recs_left = recs_left - len(candidate_reads_ids)
        
        # go to next user
        i = i + 1
    
    return recs # return your recommendations for this user_id    


# In[20]:


# Check Results


#t0 = time.time()



get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1


#t1 = time.time()

#total = t1-t0
#print(total)


# In[21]:


# Test your functions here - No need to change this code - just run this cell
assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("If this is all you see, you passed all of our tests!  Nice job!")


# `4.` Now we are going to improve the consistency of the **user_user_recs** function from above.  
# 
# * Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
# 
# 
# * Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be  what would be obtained from the **top_articles** function you wrote earlier.

# In[22]:


def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # Your code here
    #similar_users
    similarity = user_item.dot(user_item.loc[user_id])
    # count number of interactions
    num_interactions = df.groupby(["user_id"])["user_id"].count()
    
    # create a datagrame
    neighbors_df = pd.DataFrame({"neighbor_id":similarity.index,"similarity":similarity,"num_interactions":num_interactions})
    # sort the dataframe
    neighbors_df = neighbors_df.sort_values(by=["similarity","num_interactions"],ascending=False).reset_index().drop(["user_id"],axis=1)
    
    # remove user
    neighbors_df = neighbors_df[neighbors_df.neighbor_id!=user_id]
 
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    # Your code here
    
    similar_users = get_top_sorted_users(user_id)
    articles_read = get_user_articles(user_id)
 
    
    
    recs_left = m # indicator of recommendations left to make
    recs = [] # list that will store the recommendation ids
    i = 0 # counter

    # while we still have room for recommendations
    while recs_left>0:
        user = similar_users.iloc[i,i] # take the most similar user
        
        # articles that similar user has read
        candidate_reads_ids,_  = get_user_articles(user)
        
        # remove articles that the user has already seen
        candidate_reads_ids = np.setdiff1d(np.array(candidate_reads_ids), np.array(articles_read))
        
        # if the recommendations are more than the room we have
        if len(candidate_reads_ids)>recs_left:
           # add to the list
            recs.extend(candidate_reads_ids[:recs_left])
            recs_left = recs_left - len(candidate_reads_ids[:recs_left])
        
        else:
            # add to the list
            recs.append(candidate_reads_ids)
            recs_left = recs_left - len(candidate_reads_ids)
        
        # go to next user
        i = i + 1
        
        # take rec_names
        rec_names = get_article_names(recs)

    
    return recs, rec_names


# In[23]:


# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)


# `5.` Use your functions from above to correctly fill in the solutions to the dictionary below.  Then test your dictionary against the solution.  Provide the code you need to answer each following the comments below.

# In[24]:


### Tests with a dictionary of results

user1_most_sim =get_top_sorted_users(1).iloc[0,0] # Find the user that is most similar to user 1 
user131_10th_sim =get_top_sorted_users(131).iloc[9,0] # Find the 10th most similar user to user 131


# In[25]:


## Dictionary Test Here
sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim, 
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}


t.sol_5_test(sol_5_dict)


# `6.` If we were given a new user, which of the above functions would you be able to use to make recommendations?  Explain.  Can you think of a better way we might make recommendations?  Use the cell below to explain a better method for new users.

# No, because the user won't have any interactions so it is impossible to use collaborative filtering methods to make recommendations. So content-based recommendations can help us get started with making recommendations for these individuals.

# `7.` Using your existing functions, provide the top 10 recommended articles you would provide for the a new user below.  You can test your function against our thoughts to make sure we are all on the same page with how we might make a recommendation.

# In[26]:


new_user = '0.0'

# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
new_user_recs =get_top_article_ids(n=10) # Your recommendations here
new_user_recs


# In[27]:


assert set(new_user_recs) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."

print("That's right!  Nice job!")


# ### <a class="anchor" id="Content-Recs">Part IV: Content Based Recommendations </a>
# 
# Another method we might use to make recommendations is to perform a ranking of the highest ranked articles associated with some term.  You might consider content to be the **doc_body**, **doc_description**, or **doc_full_name**.  There isn't one way to create a content based recommendation, especially considering that each of these columns hold content related information.  
# 
# `1.` Use the function body below to create a content based recommender.  Since there isn't one right answer for this recommendation tactic, no test functions are provided.  Feel free to change the function inputs if you decide you want to try a method that requires more input values.  The input values are currently set with one idea in mind that you may use to make content based recommendations.  One additional idea is that you might want to choose the most popular recommendations that meet your 'content criteria', but again, there is a lot of flexibility in how you might make these recommendations.
# 
# 

# In[28]:



def tokenize(text):
    """Function to tokenize an article title

    Args:
        text (str) title

    return:
        tokens (list) a list of words

    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words]
    # remove short words
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


# In[29]:


def create_article_content_dataframe(df=df, df_content=df_content):
    '''
    INPUT:
    df - pandas dataframe describe user interaction with th articles
    df_content - pandas dataframe describe articles on the platforme
    OUTPUT:
    df_total - pandas dataframe contains all articles in the platforme
    article_content -  pandas dataframe describe the content of each
                       article on the plateforme

    Description:
    Return a pandas dataframe contains all the articles in the platforme
    with thier titles and a pandas dataframe describes the content of each
    article base on the article title.
    '''


    
    df_total = df[['article_id', 'title']].merge(df_content[["article_id","doc_full_name"]],on="article_id")
    df_total.drop_duplicates(subset=['article_id'], inplace=True)
    df_total.sort_values(by='article_id', inplace=True)
    df_total
    
    

    # initialize tf-idf vectorizer object
    vectorizer = TfidfVectorizer(tokenizer=tokenize)


    # convert sparse matrix to numpy array to view
    
    df_vec = pd.DataFrame(vectorizer.fit_transform(
                          df_total.title.values).toarray(),
                          columns=[*vectorizer.vocabulary_])
    
    
    df_articles = pd.concat([df_total.reset_index().drop("index",axis=1),df_vec],axis=1)
    

    



    return df_articles


df_articles = create_article_content_dataframe()


# In[30]:


def find_similar_articles(article_id, df_articles=df_articles):
    '''
    INPUT:
    article_id - (int) an article_id
    df_articles - (pandas dataframe) matrix of articles by tokens: 
                1's when a articles includes token, 0 otherwise
    
    OUTPUT:
    similar_articles - (list) an ordered list where the closest articles (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of articles based on the dot product
    Returns an ordered
    
    '''
    # Take article_id and tokens
    try:
        if type(article_id)==str:
            article_id = float(article_id)
        temp = df_articles[df_articles.article_id==article_id]


        # Calculate similarity matrix
        similarity = df_articles.iloc[:,3:].dot(temp.iloc[:,3:].transpose())
        # column should be named after article_id
        similarity.columns = [str(article_id)]
        # add article_id column
        similarity = pd.concat([df_articles["article_id"],similarity],axis=1)
        # Sort values by highest similarity
        similarity = similarity.sort_values(str(article_id),ascending=False)
        # exclude the article and filter by similar only
        similarity = similarity[(similarity.article_id!=article_id)&(similarity[str(article_id)]>0)]
        
        # return ids of similar articles
        similar_articles = list(similarity.article_id.astype(str))
        return similar_articles
    except:
        print("article_id does not exist in df_content")
        
       
    


# In[31]:


article_id = "0.0"

find_similar_articles(article_id)


# `2.` Now that you have put together your content-based recommendation system, use the cell below to write a summary explaining how your content based recommender works.  Do you see any possible improvements that could be made to your function?  Is there anything novel about your content based recommender?
# 

# In[32]:


print("Articles that we are not have descriptions:")
len(np.setdiff1d(df.article_id.astype(int).unique(),df_content.article_id.unique()))


# To provide content based recommendations we used the articles full name from df_content to find articles with similar words. Some of the articles in df are not part of df_content. To do so we used TfidfVectorizer. 
# 
# Some of the pros and cons.
# 
# Pros:
# - Easy to compute
# - You have some basic metric to extract the most descriptive terms in a document
# 
# Cons:
# - TF-IDF is based on the bag-of-words (BoW) model, therefore it does not capture position in text, semantics, co-occurrences in different documents, etc.
# - For this reason, TF-IDF is only useful as a lexical level feature
# - Cannot capture semantics (e.g. as compared to topic models, word embeddings)
#     
# 

# `3.` Use your content-recommendation system to make recommendations for the below scenarios based on the comments.  Again no tests are provided here, because there isn't one right answer that could be used to find these content based recommendations.
# 
# 

# In[33]:


# make recommendations for a brand new user
# a brand new user does not have any interactions so we should use ranked based recommendations
get_top_article_ids(n=10)

# make recommendations for a user who only has interacted with article id '1427.0'

find_similar_articles("1427.0") # we are not able to recommend based on an article that we do not have description


# ### <a class="anchor" id="Matrix-Fact">Part V: Matrix Factorization</a>
# 
# In this part of the notebook, you will build use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
# 
# `1.` You should have already created a **user_item** matrix above in **question 1** of **Part III** above.  This first question here will just require that you run the cells to get things set up for the rest of **Part V** of the notebook. 

# In[34]:


# Load the matrix here
#user_item_matrix = user_item.copy()
user_item_matrix = pd.read_pickle('data/user_item_matrix.p')


# In[35]:


# quick look at the matrix
user_item_matrix 


# `2.` In this situation, you can use Singular Value Decomposition from [numpy](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html) on the user-item matrix.  Use the cell to perform SVD, and explain why this is different than in the lesson.

# In[36]:


# Perform SVD on the User-Item Matrix Here

u, s, vt = np.linalg.svd(user_item_matrix)# use the built in to get the three matrices


# We can use SVD because we do not have any missing values. Our user item matrix has binary values. So no interactions would have 0 as values.

# `3.` Now for the tricky part, how do we choose the number of latent features to use?  Running the below cell, you can see that as the number of latent features increases, we obtain a lower error rate on making predictions for the 1 and 0 values in the user-item matrix.  Run the cell below to get an idea of how the accuracy improves as we increase the number of latent features.

# In[37]:


num_latent_feats = np.arange(10,700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)
    
    
plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');


# `4.` From the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations.  Instead, we might split our dataset into a training and test set of data, as shown in the cell below.  
# 
# Use the code from question 3 to understand the impact on accuracy of the training and test sets of data with different numbers of latent features. Using the split below: 
# 
# * How many users can we make predictions for in the test set?  
# * How many users are we not able to make predictions for because of the cold start problem?
# * How many articles can we make predictions for in the test set?  
# * How many articles are we not able to make predictions for because of the cold start problem?

# In[38]:


df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    # Your code here
    user_item_train  = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    test_idx = list(user_item_test.index)
    test_arts =  list(user_item_test.columns)
    
    
    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)


# In[39]:


# Replace the values in the dictionary below
a = 662 
b = 574 
c = 20 
d = 0 


sol_4_dict = {
    'How many users can we make predictions for in the test set?': c,# letter here, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?':a, # letter here, 
    'How many articles can we make predictions for in the test set?':b, # letter here,
    'How many articles in the test set are we not able to make predictions for because of the cold start problem?':d # letter here
}

t.sol_4_test(sol_4_dict)


# In[40]:


# users in train
user_item_train.shape


# In[41]:


# users in test
user_item_test.shape


# In[42]:


# how many are in the test and not in the train
len(np.setdiff1d(user_item_test.index, user_item_train.index))


# In[43]:


# items 
len(np.setdiff1d(user_item_test.columns, user_item_train.columns))


# `5.` Now use the **user_item_train** dataset from above to find **U**, **S**, and **V** transpose using SVD. Then find the subset of rows in the **user_item_test** dataset that you can predict using this matrix decomposition with different numbers of latent features to see how many features makes sense to keep based on the accuracy on the test data. This will require combining what was done in questions `2` - `4`.
# 
# Use the cells below to explore how well SVD works towards making predictions for recommendations on the test data.  

# In[44]:


# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(user_item_train, full_matrices=0)


# In[45]:


# users that exist in test
u_test = u_train[user_item_train.index.isin(test_idx),:]

# articls exist in test
vt_test = vt_train[:,user_item_train.columns.isin(test_arts)]

# user_item_test that exist in train
user_item_test_intersection = user_item_test[user_item_test.index.isin(set(user_item_train.index).intersection(user_item_test.index))]


# In[48]:


num_latent_feats = np.arange(10,700+10,20)
sum_train_errs = []
sum_test_errs = []

# for each latent feature
for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new, u_test_new, vt_test_new = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :], u_test[:,:k], vt_test[:k,:]
    
    # take dot product for predictions
    user_item_train_pred = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # test predictions
    user_item_test_pred = np.around(np.dot(np.dot(u_test_new, s_new), vt_test_new))
    
    # compute error for each training prediction to actual value
    diffs_train = user_item_train - user_item_train_pred
    
    # compute error for each testing prediction to actual value
    diffs_test = user_item_test_intersection - user_item_test_pred

    # total errors and keep track of them
    train_err = np.sum(np.sum(np.abs(diffs_train)))
    test_err = np.sum(np.sum(np.abs(diffs_test)))

    sum_train_errs.append(train_err)
    sum_test_errs.append(test_err)
    
plt.plot(num_latent_feats, 1 - np.array(sum_test_errs)/(user_item_test_intersection.shape[0] * user_item_test.shape[1]), label = 'Testing');
plt.plot(num_latent_feats, 1 - np.array(sum_train_errs)/(user_item_train.shape[0] * user_item_test.shape[1]), label = 'Training');
plt.legend()
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');


# `6.` Use the cell below to comment on the results you found in the previous question. Given the circumstances of your results, discuss what you might do to determine if the recommendations you make with any of the above recommendation systems are an improvement to how users currently find articles? 

# The more latent features we use it seems we overfit the model which makes test accuracy drop. We also still can't use the model to make recommendations to new users (we need to use the ranked based approach mentioned above). Our recommendation system can definitely  has room for improvement but need to explore other ML techniques.

# <a id='conclusions'></a>
# ### Extras
# Using your workbook, you could now save your recommendations for each user, develop a class to make new predictions and update your results, and make a flask app to deploy your results.  These tasks are beyond what is required for this project.  However, from what you learned in the lessons, you certainly capable of taking these tasks on to improve upon your work here!
# 
# 
# ## Conclusion
# 
# > Congratulations!  You have reached the end of the Recommendations with IBM project! 
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations! 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Recommendations_with_IBM.ipynb'])

