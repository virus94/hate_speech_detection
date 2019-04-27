from psaw import PushshiftAPI

# Basic libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


def scrape_data(subreddit):
	api = PushshiftAPI()
	scrape_list = list(api.search_submissions(subreddit=subreddit,
                                filter=['title', 'subreddit', 'num_comments', 'author', 'subreddit_subscribers', 'score', 'domain','created_utc'],limit=15000))

	#print(scrape_list)
	clean_scrape_lst = []
	for i in range(len(scrape_list)):
		scrape_dict = {}
		scrape_dict['subreddit'] = scrape_list[i][5]
		scrape_dict['author'] = scrape_list[i][0]
		scrape_dict['domain'] = scrape_list[i][2]
		scrape_dict['title'] = scrape_list[i][7]
		scrape_dict['num_comments'] = scrape_list[i][3]
		scrape_dict['score'] = scrape_list[i][4]
		scrape_dict['timestamp'] = scrape_list[i][1]
		clean_scrape_lst.append(scrape_dict)

    	# Show number of subscribers
	print(subreddit, 'subscribers:',scrape_list[1][6])
    	# Return list of scraped data
	return clean_scrape_lst
"""
#scrape_data('theonion')
# Call function and create DataFrame
df_not_onion = pd.DataFrame(scrape_data('nottheonion'))

# Save data to csv
df_not_onion.to_csv('data/not_onion.csv')

# Shape of DataFrame
print(f'df_not_onion shape: {df_not_onion.shape}')

# Show head
print(df_not_onion.head())

df_onion = pd.DataFrame(scrape_data('theonion'))

# Save data to csv
df_onion.to_csv('data/the_onion.csv')

# Shape of DataFrame
print(f'df_onion shape: {df_onion.shape}')

# Show head
print(df_onion.head()) """

# r/TheOnion DataFrame
df_onion = pd.read_csv('data/the_onion.csv')

# r/nottheonion DataFrame
df_not_onion = pd.read_csv('data/not_onion.csv')

print(df_onion.head())
print(df_not_onion.head())


def clean_data(dataframe):
 	# Drop duplicate rows
	dataframe.drop_duplicates(subset='title', inplace=True)
    
	# Remove punctation
	dataframe['title'] = dataframe['title'].str.replace('[^\w\s]',' ')

	# Remove numbers 
	dataframe['title'] = dataframe['title'].str.replace('[^A-Za-z]',' ')

	# Make sure any double-spaces are single 
	dataframe['title'] = dataframe['title'].str.replace('  ',' ')
	dataframe['title'] = dataframe['title'].str.replace('  ',' ')

	# Transform all text to lowercase
	dataframe['title'] = dataframe['title'].str.lower()
    
	print("New shape:", dataframe.shape)
	return dataframe.head()

clean_data(df_onion)
clean_data(df_not_onion)
print(pd.DataFrame([df_onion.isnull().sum(),df_not_onion.isnull().sum()], index=["TheOnion","notheonion"]))

df_onion['timestamp'] = pd.to_datetime(df_onion['timestamp'], unit='s')
df_not_onion['timestamp'] = pd.to_datetime(df_not_onion['timestamp'], unit='s')

# Show date-range of posts scraped from r/TheOnion and r/nottheonion
print("TheOnion start date:", df_onion['timestamp'].min())
print("TheOnion end date:", df_onion['timestamp'].max())
print("nottheonion start date:", df_not_onion['timestamp'].min())
print("nottheonion end date:", df_not_onion['timestamp'].max())

def bar_plot(x, y, title, color):    
    
	# Set up barplot 
	plt.figure(figsize=(9,5))
	g=sns.barplot(x, y, color = color)    
	ax=g

	# Label the graph
	plt.title(title, fontsize = 15)
	plt.xticks(fontsize = 10)

	# Enable bar values
	# Code modified from http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
	# create a list to collect the plt.patches data
	totals = []

	# find the values and append to list
	for p in ax.patches:
		totals.append(p.get_width())

		# set individual bar lables using above list
		total = sum(totals)

	# set individual bar lables using above list
	for p in ax.patches:
		# get_width pulls left or right; get_y pushes up or down
		ax.text(p.get_width()+.3, p.get_y()+.38, int(p.get_width()), fontsize=10)

	plt.show()

# Set x values: # of posts 
df_onion_authors = df_onion['author'].value_counts() 
df_onion_authors = df_onion_authors[df_onion_authors > 100].sort_values(ascending=False)

# Set y values: Authors 
df_onion_authors_index = list(df_onion_authors.index)

# Call function





df_not_onion_authors = df_not_onion['author'].value_counts() 
df_not_onion_authors = df_not_onion_authors[df_not_onion_authors > 100].sort_values(ascending=False)

# Set y values: Authors
df_not_onion_authors_index = list(df_not_onion_authors.index)

# Call fun# Set x values: # of postsction
#bar_plot(df_not_onion_authors.values, df_not_onion_authors_index, 'Most Active Authors: r/nottheonion','b') 
#bar_plot(df_onion_authors.values, df_onion_authors_index, 'Most Active Authors: r/TheOnion', 'r')

df = pd.concat([df_onion[['subreddit', 'title']], df_not_onion[['subreddit', 'title']]], axis=0)

#Reset the index
df = df.reset_index(drop=True)

# Preview head of df to show 'TheOnion' titles appear

df["subreddit"] = df["subreddit"].map({"nottheonion": 0, "TheOnion": 1})

# Print shape of df
#print(df.shape)

# Preview head of df to show 1s
#df.head(2)

#print(df.head(2)) 
#print("\n\n")
#print(df.tail(2))


# Set variables to show TheOnion Titles
mask_on = df['subreddit'] == 1
df_onion_titles = df[mask_on]['title']

# Instantiate a CountVectorizer
cv1 = CountVectorizer(stop_words = 'english')

# Fit and transform the vectorizer on our corpus
onion_cvec = cv1.fit_transform(df_onion_titles)

# Convert onion_cvec into a DataFrame
onion_cvec_df = pd.DataFrame(onion_cvec.toarray(),
                   columns=cv1.get_feature_names())

# Inspect head of Onion Titles cvec
print(onion_cvec_df.shape)


# Set variables to show NotTheOnion Titles
mask_no = df['subreddit'] == 0
df_not_onion_titles = df[mask_no]['title']

# Instantiate a CountVectorizer
cv2 = CountVectorizer(stop_words = 'english')

# Fit and transform the vectorizer on our corpus
not_onion_cvec = cv2.fit_transform(df_not_onion_titles)

# Convert onion_cvec into a DataFrame
not_onion_cvec_df = pd.DataFrame(not_onion_cvec.toarray(),
                   columns=cv2.get_feature_names())

# Inspect head of Not Onion Titles cvec
print(not_onion_cvec_df.shape)

onion_wc = onion_cvec_df.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(5)

# Call function
# bar_plot(onion_top_5.values, onion_top_5.index, 'Top 5 unigrams on r/TheOnion','r') 


# Set up variables to contain top 5 most used words in Onion
nonion_wc = not_onion_cvec_df.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(5)

# Call function
#bar_plot(nonion_top_5.values, nonion_top_5.index, 'Top 5 unigrams on r/nottheonion','b')

# Create list of unique words in top five
not_onion_5_set = set(nonion_top_5.index)
onion_5_set = set(onion_top_5.index)

# Return common words
common_unigrams = onion_5_set.intersection(not_onion_5_set)
print(common_unigrams)

# Set variables to show TheOnion Titles
mask = df['subreddit'] == 1
df_onion_titles = df[mask]['title']

# Instantiate a CountVectorizer
cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))

# Fit and transform the vectorizer on our corpus
onion_cvec = cv.fit_transform(df_onion_titles)

# Convert onion_cvec into a DataFrame
onion_cvec_df = pd.DataFrame(onion_cvec.toarray(),
                   columns=cv.get_feature_names())

# Inspect head of Onion Titles cvec
print(onion_cvec_df.shape)

# Set variables to show NotTheOnion Titles
mask = df['subreddit'] == 0
df_not_onion_titles = df[mask]['title']

# Instantiate a CountVectorizer
cv = CountVectorizer(stop_words = 'english', ngram_range=(2,2))

# Fit and transform the vectorizer on our corpus
not_onion_cvec = cv.fit_transform(df_not_onion_titles)

# Convert onion_cvec into a DataFrame
not_onion_cvec_df = pd.DataFrame(not_onion_cvec.toarray(),
                   columns=cv.get_feature_names())

# Inspect head of Not Onion Titles cvec
print(not_onion_cvec_df.shape)


# Set up variables to contain top 5 most used bigrams in r/TheOnion
onion_wc = onion_cvec_df.sum(axis = 0)
onion_top_5 = onion_wc.sort_values(ascending=False).head(5)

# Call function
#bar_plot(onion_top_5.values, onion_top_5.index, 'Top 5 bigrams on r/TheOnion','r')


# Set up variables to contain top 5 most used bigrams in r/nottheonion
nonion_wc = not_onion_cvec_df.sum(axis = 0)
nonion_top_5 = nonion_wc.sort_values(ascending=False).head(5)

# Call function
#bar_plot(nonion_top_5.values, nonion_top_5.index, 'Top 5 bigrams on r/nottheonion','b')

not_onion_5_list = set(nonion_top_5.index)
onion_5_list = set(onion_top_5.index)

# Return common words
common_bigrams = onion_5_list.intersection(not_onion_5_list)
print(common_bigrams)

































