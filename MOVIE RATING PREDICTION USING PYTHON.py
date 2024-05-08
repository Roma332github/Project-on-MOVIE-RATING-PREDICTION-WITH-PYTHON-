#!/usr/bin/env python
# coding: utf-8

# # MOVIE RATING PREDICTION Using Python👩🏻‍💻📈💼
# 
# 
# 
# 
# 
# 

# ![imdb%20logo2.jpg](attachment:imdb%20logo2.jpg)

# ![movie%20poster1.webp](attachment:movie%20poster1.webp)

# # Movie Rating Prediction: Data Science Project 📊📈💼
# 
# 
# ## In this Data Science Project on Movie Rating Prediction , every dataset is extracted from IMDb.com of all the Indian movies.
# 
# ## We'll build a movie rating prediction model using machine learning techniques. We'll use a dataset containing information about movies, such as Genres, Director , Actor and other features to predict movie ratings.
# 
# 
# # Let's Begin✅💻
# 
# 
# ### In this project focused on IMDb movie rating prediction, key features such as genre, director, and actors play pivotal roles in determining a movie's success. The project begins with data collection from IMDb, capturing comprehensive movie attributes including genres, directorial credits, and cast members.
# 
# ### Data cleaning and preprocessing involve handling missing values and transforming categorical features like genre into numerical representations using techniques such as one-hot encoding. Exploratory data analysis (EDA) delves into understanding the distribution of movie ratings across different genres and the influence of directors and actors on ratings.
# 
# ## Step 1: Data Collection📁 
# 
# ### Explanation: Obtaining a comprehensive dataset containing movie-related information.
# 
# ### The given dataset includes details of features about movies such as name , year ,duration, genre , rating , votes ,director and actors.
# 
# ## Step 2: Data Cleaning and Preprocessing📋
# 
# ### Explanation: Clean and preprocess the dataset to prepare it for analysis.
# 
# 
# ### Handeling missing values: Remove or impute missing values in the dataset, especially in critical columns like ratings.
# ### Feature selection: Choose relevant features for the prediction task here, genre, director and actors.
# ### Data transformation: Convert categorical variables like genres into numerical representations .
# ### Spliting data: Divide the dataset into features (X) and the target variable (y) which is the movie rating.
# 
# ## Step 3: Exploratory Data Analysis (EDA) 📈📉
# 
#  ### Explanation:Explore and visualize the dataset to gain insights.
# 
# ### Analyze the distribution of movie ratings using histograms or density plots.Identify correlations between features and ratings using scatter plots or correlation matrices.Visualize categorical variables e.g. genres to understand their impact on ratings.
# 
# ## Step 4: Feature Engineering 📝
# 
# ### Explanation: Create new features to enhance the predictive power of the model.
# 
# ### Extracting useful information from existing features e.g., extracting year from release date. Generating additional features like director or actor popularity based on historical data.Scaling or normalizing numerical features to ensure all features contribute equally to the model.
# 
# ## Step 5: Model Selection and Training 💻
# 
# ### Explanation: Choose an appropriate machine learning model and train it on the dataset.
# 
# ### Selecting a regression model suitable for predicting continuous ratings  e.g., linear regression, random forest regression.Spliting the dataset into training and testing sets to evaluate model performance.Training the model using the training dataset and validate it using the testing dataset.
# 
# ## Step 6: Model Evaluation 📌
# ### Explanation: Evaluate the model's performance using appropriate metrics.
# 
# ### Using evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), or R-squared to quantify the model's accuracy. Comparing the predicted ratings with actual ratings to assess the model's effectiveness in predicting movie ratings.
# 
# ## Step 7: Model Deployment 📉
# 
# ### Explanation: Deploy the trained model for making predictions on new data.
# 
# ### Once the model is trained and evaluated, deploy it to a production environment. We can use the deployed model to predict movie ratings for new movies based on their attributes e.g., genres, cast.
# 
# 
# ## By following these structured steps, we can develop a robust movie rating prediction model using data science techniques. Each step is crucial in the data science lifecycle, from data collection and cleaning to model training and deployment.

# ## Let's See This Practically Through Python 👩🏻‍💻

# # Importing Required Libraries

# In[1]:


import numpy as np


# ## Reading the Data

# In[2]:


import pandas as pd
df = pd.read_csv('Moviedataset.csv', encoding='latin1')


# ## Data Cleaning and Preprocessing

# ### Defining the Dataframe

# In[3]:


df


# ### Describing the head of the Dataframe

# In[4]:


df.head()


# ### Describing the tail of Dataframe

# In[5]:


df.tail()


# ### Defining shape and size of Dataframe

# In[6]:


df.shape


# ### Describing the Dataframe

# In[7]:


df.describe()


# ### Getting the information of the column, non-null count and datatype 

# In[8]:


df.info()


# In[9]:


df.dtypes


# ### Finding the Null Values

# In[10]:


df.isnull()


# #### From this it could be concluded that the Dataframe is containing some null values.
# 
# #### We can calculate that how many null values are in the Dataframe using isnull().sum() function.

# In[11]:


df.isnull().sum()


# #### Hence there are a large number of null values in the dataframe.
# 
# #### The dropna(inplace=True) function call is used to remove rows or columns depending on the axis parameter that contain missing values or None values from a Dataframe.
# 
# ### Let's see the head of dataframe with removed null values 

# In[12]:


df.dropna(inplace=True)


# In[13]:


df.head()


# ### Let's Check if the null values are fully removed from the Dataframe

# In[14]:


df.isnull().sum()


# ### Hence All the null values are removed from the Dataframe

# ### Data Transformation

# In[16]:


df['Year'] = df['Year'].str.extract('(\d+)')  # Extract numeric part of the string
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Convert to numeric


# In[17]:


df['Duration'] = df['Duration'].str.extract('(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')


# In[18]:


df["Year"].head()


# #### Hence we get the Year in integer format. 

# In[20]:


df["Genre"].head()


# #### There are multiple values in Genre column, Hence we can use   .str.split(',',expand=True)   function to split the column Genre into multiple columns based on a delimiter (in this case ',' comma)

# In[21]:


df['Genre'].str.split(',',expand=True)


# In[23]:


genres=df['Genre'].str.split(',',expand=True)
genres.head(5)


# #### We create an empty dictionary (genre_counts) to store genre counts.
# #### It loops through all the genres extracted from the flattened DataFrame (genres.values.flatten()).
# #### For each genre (ignoring None values), it checks if the genre already exists in genre_counts.
# #### If the genre exists, it increments its count by 1.
# #### If the genre doesn't exist yet, it adds the genre to genre_counts with an initial count of 1.
# #### After counting all genre occurrences, it sorts the genre_counts dictionary based on genre names.
# #### It then iterates through the sorted genre counts (genreCounts) and prints each genre along with its count.
# 

# In[24]:


genre_counts = {}
for genre in genres.values.flatten():
    if genre is not None:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

genereCounts = {genre: count for genre, count in sorted(genre_counts.items())}
for genre, count in genereCounts.items():
    print(f"{genre}: {count}")


# #### Hence We are able to count the Genre values

# #### To show the frequency of different movie genres present in the 'Genre' column of a DataFrame-
# #### We can count how many times each genre appears in the 'Genre' column.
# #### For example, if 'Action' appears 250 times, 'Drama' appears 200 times, 'Comedy' appears 180 times, and so on.
# #### After calculating genre counts, it retrieves the top 5 most frequent genres.

# In[25]:


genresPie = df['Genre'].value_counts()
genresPie.head(5)


# In[26]:


genrePie = pd.DataFrame(list(genresPie.items()))
genrePie = genrePie.rename(columns={0: 'Genre', 1: 'Count'})
genrePie.head(5)


# In[27]:


df['Votes'] = df['Votes'].str.replace(',', '').astype(int)
df["Votes"].head(5)


# In[28]:


df["Director"].nunique() #nunique() function in pandas is used to count the number of unique (distinct) values in a Series or DataFrame column.


# In[29]:


df["Director"].value_counts()


# #### Hence there are 3 values of Actors we can combine the all three actor names in a column. 

# In[31]:


actors = pd.concat([df['Actor 1'], df['Actor 2'], df['Actor 3']]).dropna().value_counts()
actors.head()


# ## Exploratory Data Analysis (EDA)
# 
# 
# #### Exploratory Data Analysis (EDA) is a fundamental step in the data science lifecycle, providing valuable insights into the dataset and guiding subsequent data preprocessing, modeling, and analysis tasks. By leveraging descriptive statistics, visualizations, and analytical techniques, data scientists can uncover hidden patterns and relationships within the data, leading to more informed decision-making and impactful data-driven solutions.
# 
# ## Data Visualization
# 
# ### Importing Libraries 

# In[34]:


import seaborn as sb
import plotly.express as px
import matplotlib.pyplot as mpl
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud


# ### Seaborn
# 
# #### Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for creating attractive and informative statistical graphics. It is designed to work seamlessly with Pandas DataFrames and arrays, making it easy to visualize data directly from these data structures.
# 
# #### Lineplot
# 
# #### Creating a line plot (lineplot) to visualize how the count of movie releases varies over the years. It counts the number of movies released each year, sorts the counts by year, and then plots this data as a line plot. The x-axis represents the years, and the y-axis represents the count of movie releases.
# #### The plot title is set to "Annual Movie Release Counts Over Time".The x-axis tick marks are positioned and labeled with years, spaced every 5 years.The x-axis is labeled as "Years", and the y-axis is labeled as "Count".
# 

# In[35]:


sb.set(style = "ticks", font = "Times New Roman")


# In[36]:


ax = sb.lineplot(data=df['Year'].value_counts().sort_index())
tick_positions = range(min(df['Year']), max(df['Year']) + 1, 5)
ax.set_title("Annual Movie Release Counts Over Time")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_positions, rotation = 90)
ax.set_xlabel("Years")
ax.set_ylabel("Count")
mpl.show()


# #### df['Year'].value_counts(): Counts the occurrences of each unique year in the 'Year' column of the DataFrame df.sort_index(): Sorts the counts based on the index (years in this case) in ascending order. And sb.lineplot(): Creates a line plot using Seaborn (sb) with the sorted year counts.
# 
# #### min(df['Year']): Finds the minimum year value in the 'Year' column, max(df['Year']): Finds the maximum year value in the 'Year' column, range(min(df['Year']), max(df['Year']) + 1, 5): Generates a range of tick positions for the x-axis, starting from the minimum year to the maximum year, with steps of 5 years.
# 
# #### ax.set_title("Annual Movie Release Counts Over Time"): Sets the title of the plot to "Annual Movie Release Counts Over Time", ax.set_xticks(tick_positions): Sets the x-axis tick positions based on the tick_positions range defined earlier, ax.set_xticklabels(tick_positions, rotation=90): Sets the x-axis tick labels to the years (from tick_positions) and rotates them by 90 degrees for better readability, ax.set_xlabel("Years"): Sets the label for the x-axis to "Years", ax.set_ylabel("Count"): Sets the label for the y-axis to "Count", mpl.show(): Displays the plot using Matplotlib (mpl).
# 

# #### Boxplot

# In[37]:


ax = sb.boxplot(data=df, y='Year')
ax.set_ylabel('Year')
ax.set_title('Box Plot of Year')
mpl.show()


# #### The box plot (sb.boxplot) is used to visualize the distribution and central tendency of the 'Year' column in the DataFrame df. It provides insights into the spread of movie release years, including the minimum, maximum, median (middle line of the box), and quartiles (edges of the box) of the distribution. By creating a box plot, you can quickly identify outliers (data points outside the whiskers) and assess the overall variability of movie release years. 

# In[38]:


ax = sb.lineplot(data=df.groupby('Year')['Duration'].mean().reset_index(), x='Year', y='Duration')
tick_positions = range(min(df['Year']), max(df['Year']) + 1, 5)
ax.set_title("Average Movie Duration Trends Over the Years")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_positions, rotation = 90)
ax.set_xlabel("Years")
ax.set_ylabel('Average Duration(in minutes)')
mpl.show()


# #### The line plot helps to understand how the average duration of movies has evolved over different years. Trends in movie duration can be observed, indicating potential changes in filmmaking styles or audience preferences over time.

# In[39]:


ax = sb.boxplot(data=df, y='Duration')
ax.set_title("Box Plot of Average Movie Durations")
ax.set_ylabel('Average Duration(in minutes)')
mpl.show()


# #### We can see that in Box plot of Average Movie duration outliers are present .
# #### We can perform outlier removal based on the Interquartile Range (IQR) method for the 'Duration' column in the DataFrame df, ensuring that only data within a certain range (defined by the quartiles and IQR) is retained for further analysis or processing.

# In[43]:


Q1 = df['Duration'].quantile(0.25)
Q3 = df['Duration'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Duration'] >= lower_bound) & (df['Duration'] <= upper_bound)]
df.head()


# ### Matplotlib
# 
# #### Matplotlib is a widely used plotting library in Python that enables users to create a variety of high-quality, customizable visualizations for data exploration, analysis, and presentation. It provides a flexible and extensive toolkit for generating static plots in various formats.
# 
# #### Wordcloud
# 
# #### A word cloud in Matplotlib is a visual representation of text data where words are displayed in different sizes based on their frequency. It's created using the WordCloud library, which arranges words in a graphical manner.
# #### word clouds offer a simple and intuitive way to visually summarize textual data.

# In[44]:


genre_counts = df['Genre'].str.split(', ', expand=True).stack().value_counts()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genre_counts)

mpl.figure(figsize=(10, 6))
mpl.imshow(wordcloud, interpolation='bilinear')
mpl.axis('off')
mpl.title('Genre Word Cloud')
mpl.show()


# #### wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genre_counts) This initializes a WordCloud object with specific parameters: width=800, height=400: Sets the dimensions of the word cloud visualization, background_color='white': Sets the background color of the word cloud to white, generate_from_frequencies(genre_counts): Uses the genre_counts (genre frequency) to generate the word cloud, where the size of each genre word is based on its frequency. 
# #### mpl.figure(figsize=(10, 6)): Sets the figure size for plotting the word cloud (width=10 inches, height=6 inches), mpl.imshow(wordcloud, interpolation='bilinear'): Displays the word cloud using Matplotlib (mpl), with bilinear interpolation for smoother image rendering, mpl.axis('off'): Turns off the axis (x and y) labels and ticks for cleaner visualization, mpl.title('Genre Word Cloud'): Sets the title of the plot to 'Genre Word Cloud', mpl.show(): Finally, shows the word cloud plot.

# #### Barplot

# In[54]:


genreLabels = sorted(genereCounts.keys())
genreCounts = sorted(genereCounts.values())
ax = sb.barplot(x = genreLabels, y = genreCounts, color='purple')
ax.set_xticklabels(labels=genreLabels, rotation = 90)
mpl.xlabel('Genres')  # Label for x-axis
mpl.ylabel('Counts')  # Label for y-axis
mpl.show()


# #### A bar plot using Seaborn (sb.barplot), customizes the x-axis labels to show genre names with rotated text for better presentation, and then displays the plot using Matplotlib (mpl.show()). The resulting visualization will show a bar chart of genre counts, with each bar representing a different genre labeled on the x-axis. 

# #### PieChart

# In[102]:


genrePie.loc[genrePie['Count'] < 50, 'Genre'] = 'Other'
ax = px.pie(genrePie, values='Count', names='Genre', title='More than one Genre of movies in Indian Cinema')
ax.show()


# #### In this code  we first modifiy the genrePie DataFrame by replacing genre categories with fewer than 50 counts as 'Other'. This consolidation simplifies the visualization by grouping less common genres together.
# 
# #### Next, we use Plotly Express (px) to create a pie chart (px.pie) based on the updated DataFrame. The chart represents different genres of movies in Indian cinema, where each slice of the pie corresponds to a genre, and the size of each slice represents the count of movies in that genre.
# 
# #### Finally, the code displays the pie chart, allowing you to interactively explore and visualize the distribution of movie genres in Indian cinema based on the provided data. The title of the chart helps to convey the purpose of the visualization.

# #### Histogram
# 

# In[56]:


ax = sb.histplot(data = df, x = "Rating", bins = 20, kde = True)
ax.set_xlabel('Rating')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Movie Ratings')
mpl.show()


# #### The histogram visually represents the distribution of movie ratings across the dataset.
# #### Each bar in the histogram represents a range of ratings (bin), and the height of each bar indicates the frequency (count) of movies with ratings falling within that range.
# #### The KDE curve overlaid on the histogram provides a smoothed representation of the rating distribution, highlighting any patterns or trends in how ratings are distributed

# #### Barplot

# In[57]:


ax = sb.boxplot(data=df, y='Rating')
ax.set_ylabel('Rating')
ax.set_title('Box Plot of Movie Ratings')
mpl.show()


# #### The box plot summarizes the distribution of movie ratings based on key statistical measures:
# #### The thick line inside the box represents the median (50th percentile) of the ratings.
# #### The box itself spans the interquartile range (IQR), which covers the middle 50% of the ratings.
# #### The "whiskers" extend to the minimum and maximum values within a calculated range, excluding outliers.
# #### Points beyond the whiskers are considered outliers and are plotted individually.
# 
# ### Hence We can see that there are 'Outliers' in Movie Ratings.
# 
# #### The Interquartile Range (IQR) method  is used to identify and remove potential outliers from the 'Rating' column in the DataFrame df. By calculating quartiles (Q1 and Q3) and then defining outlier bounds based on these quartiles and the IQR, the code filters the DataFrame to exclude rows with ratings that are considered outliers. The resulting DataFrame (df) contains only the data points within a reasonable range of ratings, helping to clean and prepare the data for further analysis or visualization.

# In[65]:


Q1 = df['Rating'].quantile(0.25)
Q3 = df['Rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Rating'] >= lower_bound) & (df['Rating'] <= upper_bound)]
df.head()


# #### After removing Outliers the Dataframe.
# 
# #### Lineplot 

# In[59]:


rating_votes = df.groupby('Rating')['Votes'].sum().reset_index()
mpl.figure(figsize=(10, 6))
ax_line_seaborn = sb.lineplot(data=rating_votes, x='Rating', y='Votes', marker='o')
ax_line_seaborn.set_xlabel('Rating')
ax_line_seaborn.set_ylabel('Total Votes')
ax_line_seaborn.set_title('Total Votes per Rating')
mpl.show()


# #### The line plot illustrates how the total number of votes (y-axis) varies across different movie ratings (x-axis).
# #### Each data point on the line represents a specific rating category, showing the cumulative votes received for movies within that rating range.
# #### The use of markers (o) highlights individual data points, making it easier to visualize the distribution and trends in voting patterns across different ratings.

# In[68]:


director = df['Director'].value_counts()

mpl.figure(figsize=(10, 6))
ax = sb.barplot(x=director.head(20).index, y=director.head(20).values, palette='viridis')
ax.set_xlabel('Director')
ax.set_ylabel('Frequency of Movies')
ax.set_title('Top 20 Directors by Frequency of Movies')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
mpl.show()


# #### This bar graph represents the calculated  frequency of movies directed by each director (director Series), creates a bar plot to visualize the top 20 directors by movie frequency using Seaborn (sb), customizes plot labels and title for clarity, rotates x-axis labels for readability, and then displays the plot using Matplotlib (mpl). The resulting visualization helps identify the most prolific directors based on the number of movies they have directed.

# In[69]:


mpl.figure(figsize=(10, 6))
ax = sb.barplot(x=actors.head(20).index, y=actors.head(20).values, palette='viridis')
ax.set_xlabel('Actors')
ax.set_ylabel('Total Number of Movies')
ax.set_title('Top 20 Actors with Total Number of Movies')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
mpl.show()


# #### This is a bar plot using Seaborn and Matplotlib to visualize the total number of movies for the top 20 actors. The resulting plot provides a clear representation of actor names and their corresponding movie counts, presented in an informative and visually appealing manner. 

# In[70]:


df["Actor"] = df['Actor 1'] + ', ' + df['Actor 2'] + ', ' + df['Actor 3']
df["Directors"] = df['Director'].astype('category').cat.codes
df["Genres"] = df['Genre'].astype('category').cat.codes
df["Actors"] = df['Actor'].astype('category').cat.codes
df.head(5)


# #### Concatenation of Actors: The "Actor" column combines actor names from three separate columns ('Actor 1', 'Actor 2', 'Actor 3') into a single string, facilitating easier analysis or visualization of actor combinations.
# 
# #### Categorical Encoding:  
# #### Directors: Converts director names ('Director' column) into numerical codes for machine learning models or analysis tasks that require numeric inputs.
# #### Genres: Converts genre categories ('Genre' column) into numeric codes for the same purpose.
# #### Actors: Converts concatenated actor names ('Actor' column) into numeric codes, similarly enabling numerical representations for categorical actor data.
# #### "  Converting categorical data into numeric codes allows for more efficient processing and analysis in machine learning "

# In[71]:


ax = sb.boxplot(data=df, y='Genres')
ax.set_ylabel('Genres')
ax.set_title('Box Plot of Genres')
mpl.show()


# #### Box Plot: A box plot is a graphical representation that displays the distribution and statistical summary of a numerical or categorical variable. It shows key statistics such as median, quartiles, and potential outliers.
# #### Categorical Variable ('Genres'): In this context, the box plot visualizes the distribution of the categorical variable 'Genres', which likely represents different movie genres (e.g., Action, Comedy, Drama, etc.).
# #### Interpreting the Box Plot: The box in the plot represents the interquartile range (IQR), with the median value shown as a line inside the box.The "whiskers" extend to show the range of the data, excluding outliers (represented as individual points beyond the whiskers). Outliers can provide insights into unusual or extreme values within specific genres. 

# In[72]:


Q1 = df['Genres'].quantile(0.25)
Q3 = df['Genres'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Genres'] >= lower_bound) & (df['Genres'] <= upper_bound)]
df.head(5)


# #### Interquartile Range (IQR): The IQR is a measure of statistical dispersion that describes the spread of the middle 50% of the data. It is used to identify and filter out potential outliers.
# #### Outlier Detection: The calculated lower_bound and upper_bound define a range beyond which data points are considered potential outliers.
# #### DataFrame Filtering: The DataFrame df is modified to retain only the rows where genre values are within the acceptable range defined by the IQR-based bounds.
# #### Data Integrity: Filtering based on IQR helps in cleaning and preprocessing the data by removing extreme values that may skew analysis or visualization. 
# ####  the Interquartile Range (IQR) method to identify and remove potential outliers from a DataFrame based on the 'Genres' column. By applying statistical measures to define outlier bounds, the code ensures that the dataset contains genre data within a reasonable and representative range for analysis or modeling purposes.

# In[73]:


ax = sb.boxplot(data=df, y='Directors')
ax.set_ylabel('Directors')
ax.set_title('Box Plot of Directors')
mpl.show()


# #### Understanding the Plot: 1.)The vertical axis (y='Directors') represents the categorical variable being analyzed, which in this case is director names. 2.)The box in the plot represents the interquartile range (IQR) of director names, with the median value shown as a line inside the box. 3.)The "whiskers" extend to show the range of director names, excluding outliers (represented as individual points beyond the whiskers).
# #### Interpreting the Plot: The box plot helps visualize the distribution of directors based on the dataset. It provides insights into the variability and central tendency of director names, highlighting potential outliers or extreme values.
# 

# In[74]:


df.head()


# In[75]:


ax = sb.boxplot(data=df, y='Actors')
ax.set_ylabel('Actors')
ax.set_title('Box Plot of Actors')
mpl.show()


# #### Understanding the Plot: 1.)The vertical axis (y='Actors') represents the categorical variable being analyzed, which in this case is actor names. 2.)The box in the plot represents the interquartile range (IQR) of actor names, with the median value shown as a line inside the box. 3.)The "whiskers" extend to show the range of actor names, excluding outliers (represented as individual points beyond the whiskers).
# #### Interpreting the Plot: The box plot helps visualize the distribution of actors based on the dataset. It provides insights into the variability and central tendency of actor names, highlighting potential outliers or extreme values.

# In[76]:


Q1 = df['Actors'].quantile(0.25)
Q3 = df['Actors'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Actors'] >= lower_bound) & (df['Actors'] <= upper_bound)]
df.head(5)


# #### the Interquartile Range (IQR) method to identify and remove potential outliers from a DataFrame based on the 'Actors' column. By applying statistical measures to define outlier bounds, the code ensures that the dataset contains actor counts within a reasonable and representative range for analysis or modeling purposes. 

# ## Splitting The Data

# In[77]:


from sklearn.model_selection import train_test_split


# #### This line imports the train_test_split function from the model_selection module of the sklearn library (scikit-learn). This function will be used to split the dataset into training and testing sets.

# In[78]:


Input = df.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Actor'], axis=1)
Output = df['Rating']


# #### Feature Selection:
# #### The Input DataFrame is prepared by selecting specific columns (features) from the original dataset to be used as input for training the model. Unnecessary columns such as 'Name', 'Genre', and actor-related columns ('Actor 1', 'Actor 2', 'Actor 3', 'Actor') are dropped to focus on relevant features.
# #### Target Variable:
# #### The Output Series contains the target variable ('Rating'), which represents the values that the machine learning model will predict.
# #### Data Preparation:
# #### This demonstrates an essential step in preparing data for machine learning tasks, where input features and target variable are separated and formatted appropriately for model training.
# #### Data Integrity:
# #### By dropping unnecessary columns and selecting relevant features, the input data (Input) is streamlined and optimized for training a predictive model.

# In[80]:


Input.head()


# In[81]:


Output.head()


# #### This demonstrates the process of preparing input features (Input) and the target variable (Output) for training a machine learning model using the sklearn library. The Input DataFrame represents the independent variables used to predict the movie ratings (Output)

# In[82]:


x_train, x_test, y_train, y_test = train_test_split(Input, Output, test_size = 0.2, random_state = 1)


# #### Training and Testing Sets:
# #### x_train: This variable contains the input features (Input) used for training the machine learning model. x_test: This variable contains a subset of the input features (Input) used for evaluating the trained model. y_train: This variable contains the target variable (Output) corresponding to the training set (x_train), used to train the model to predict movie ratings. y_test: This variable contains the target variable (Output) corresponding to the testing set (x_test), used to evaluate the model's performance in predicting movie ratings. 
# #### Data Splitting:
# #### The train_test_split function randomly divides the dataset (Input and Output) into training and testing sets based on the specified test_size. This process ensures that the model is trained on a portion of the data (x_train, y_train) and evaluated on unseen data (x_test, y_test) to assess its generalization ability.

# # THE MODEL

# ## Introduction to Model Creation for Movie Rating Prediction 
# ### In the realm of data science and machine learning, the creation of predictive models plays a vital role in various applications, including movie rating prediction. This introduction provides an overview of the process involved in developing a machine learning model to predict movie ratings based on relevant features.
# 
# ### Objective : The primary objective of this model creation is to build a predictive algorithm that can accurately estimate the ratings of movies using certain input features. By leveraging historical movie data, we aim to train a model that can generalize well to unseen movies and make reliable predictions.
# ### Dataset : The model creation process begins with a dataset containing information about movies, such as duration, directors, genres, and actors, alongside their corresponding ratings. This dataset serves as the foundation for training and evaluating our machine learning model.
# 

# ### Importing required Libraries

# In[104]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score as score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


# #### LinearRegression (sklearn.linear_model.LinearRegression): Represents a linear regression model that fits a linear relationship between input features and target values.
# #### RandomForestRegressor (sklearn.ensemble.RandomForestRegressor): Utilizes an ensemble of decision trees to perform regression tasks, offering robustness and handling of complex relationships.
# #### GradientBoostingRegressor (sklearn.ensemble.GradientBoostingRegressor): Implements gradient boosting, a technique that builds an ensemble of weak learners (decision trees) to improve predictive performance.
# #### DecisionTreeRegressor (sklearn.tree.DecisionTreeRegressor): Constructs a regression tree to predict target values based on input features, recursively splitting data into homogenous subsets.
# #### XGBRegressor (xgboost.XGBRegressor): An implementation of gradient boosting optimized for speed and performance, offering advanced hyperparameter tuning capabilities.
# #### LGBMRegressor (lightgbm.LGBMRegressor): Leverages lightGBM, a fast and efficient gradient boosting framework, suitable for large datasets and high-dimensional features.
# #### CatBoostRegressor (catboost.CatBoostRegressor): Integrates catboost, a machine learning library that handles categorical features seamlessly and provides robust regression models.
# #### KNeighborsRegressor (sklearn.neighbors.KNeighborsRegressor): Predicts target values by averaging the values of k-nearest neighbors in the feature space, suitable for local approximation.
# #### SVR (sklearn.svm.SVR): Implements Support Vector Regression (SVR), which finds a hyperplane in a high-dimensional space to minimize prediction errors. 
# 
# #### " By importing these models, we can instantiate and use them for training and evaluating predictive models on our dataset to determine which one performs best for predicting movie ratings."

# In[98]:


def evaluate_model(y_true, y_pred, model_name):
    print("Model: ", model_name)
    print("Accuracy = {:0.2f}%".format(score(y_true, y_pred)*100))
    print("Mean Squared Error = {:0.2f}\n".format(mean_squared_error(y_true, y_pred, squared=False)))
    return round(score(y_true, y_pred)*100, 2)


# #### Evaluation Metrics:
# #### The function evaluate_model calculates and prints two important evaluation metrics for regression models: 1.)Accuracy: Typically calculated as the coefficient of determination (r2_score), which measures the proportion of the variance in the target variable that is predictable from the input features. 2.)Mean Squared Error (RMSE): Measures the average squared difference between the true and predicted values. A lower RMSE indicates a better fit of the model to the data.
# #### Model Name:
# #### The model_name parameter allows us to identify which specific model is being evaluated, which is useful for tracking and comparing results across different models.
# #### Utility of the Function: 
# #### This function simplifies the process of evaluating machine learning models by encapsulating the calculation and printing of key performance metrics into a reusable and concise block of code. It promotes code reusability and readability, enabling quick evaluation of model performance during model development and experimentation.

# In[99]:


LR = LinearRegression()
LR.fit(x_train, y_train)
lr_preds = LR.predict(x_test)

RFR = RandomForestRegressor(n_estimators=100, random_state=1)
RFR.fit(x_train, y_train)
rf_preds = RFR.predict(x_test)

DTR = DecisionTreeRegressor(random_state=1)
DTR.fit(x_train, y_train)
dt_preds = DTR.predict(x_test)

XGBR = XGBRegressor(n_estimators=100, random_state=1)
XGBR.fit(x_train, y_train)
xgb_preds = XGBR.predict(x_test)

GBR = GradientBoostingRegressor(n_estimators=100, random_state=60)
GBR.fit(x_train, y_train)
gb_preds = GBR.predict(x_test)

LGBMR = LGBMRegressor(n_estimators=100, random_state=60)
LGBMR.fit(x_train, y_train)
lgbm_preds = LGBMR.predict(x_test)

CBR = CatBoostRegressor(n_estimators=100, random_state=1, verbose=False)
CBR.fit(x_train, y_train)
catboost_preds = CBR.predict(x_test)

KNR = KNeighborsRegressor(n_neighbors=5)
KNR.fit(x_train, y_train)
knn_preds = KNR.predict(x_test)


# #### Model Initialization and Training:
# #### Several regression models are instantiated and trained using the training data (x_train, y_train). Each model is fitted to learn the relationship between the input features (x_train) and the target variable (y_train).
# #### Model Predictions: 
# #### After training, predictions are made using each trained model on the test data (x_test). The predict method is used for each model (LR, RFR, DTR, XGBR, GBR, LGBMR, CBR, KNR) to generate predictions (lr_preds, rf_preds, dt_preds, xgb_preds, gb_preds, lgbm_preds, catboost_preds, knn_preds) for the test dataset.

# In[100]:


LRScore = evaluate_model(y_test, lr_preds, "LINEAR REGRESSION")
RFScore = evaluate_model(y_test, rf_preds, "RANDOM FOREST")
DTScore = evaluate_model(y_test, dt_preds, "DECEISION TREE")
XGBScore = evaluate_model(y_test, xgb_preds, "EXTENDED GRADIENT BOOSTING")
GBScore = evaluate_model(y_test, gb_preds, "GRADIENT BOOSTING")
LGBScore = evaluate_model(y_test, lgbm_preds, "LIGHT GRADIENT BOOSTING")
CBRScore = evaluate_model(y_test, catboost_preds, "CAT BOOST")
KNNScore = evaluate_model(y_test, knn_preds, "K NEAREST NEIGHBORS")


# #### Model Selection and Configuration:
# #### Various regression models are employed, including Linear Regression, Random Forest Regression, Decision Tree Regression, XGBoost Regression, Gradient Boosting Regression, LightGBM Regression, CatBoost Regression, and K-Nearest Neighbors Regression. Each model may have different hyperparameters (n_estimators, random_state, etc.) that influence their performance and behavior.
# #### Training Process:
# #### The fit method is used to train each model (LR.fit(), RFR.fit(), DTR.fit(), etc.) on the training data (x_train, y_train). During training, the models learn patterns and relationships in the training dataset to make accurate predictions.
# #### Prediction Process:
# #### Once trained, the predict method is applied to the test data (x_test) using each model to generate predictions (lr_preds, rf_preds, dt_preds, etc.). Predictions (y_pred) are obtained for the test dataset, allowing us to evaluate the performance of each model.

# In[101]:


models = pd.DataFrame(
    {
        "MODELS": ["Linear Regression", "Random Forest", "Decision Tree", "Gradient Boosting", "Extended Gradient Boosting", "Light Gradient Boosting", "Cat Boosting", "K Nearest Neighbors"],
        "SCORES": [LRScore, RFScore, DTScore, GBScore, XGBScore, LGBScore, CBRScore, KNNScore]
    }
)
models.sort_values(by='SCORES', ascending=False)


# #### Creating a DataFrame (models):
# #### A new Pandas DataFrame named models is created to store model names and their corresponding scores. The DataFrame is constructed using a dictionary-like syntax where:
# #### "MODELS" represents a column containing names of different regression models. "SCORES" represents a  column containing corresponding scores obtained from evaluating each model.
# #### The "MODELS" column contains a list of model names. The "SCORES" column contains a list of scores (LRScore, RFScore, etc.) corresponding to each model's performance.
# #### Sorting the DataFrame:
# #### After creating the DataFrame, the sort_values method is used to sort the DataFrame based on the values in the "SCORES" column in descending order (ascending=False).
# #### This rearranges the rows of the DataFrame so that models with higher scores (better performance) appear first.
# 

# # Conclusion 📋👩🏻‍💻
# 
# ## Model Selection and Performance 🏆 :
# ### 1.)Light Gradient Boosting: Achieved the highest score of 42.65, indicating superior predictive performance among the models evaluated. Light Gradient Boosting methods often excel in handling large datasets efficiently while maintaining good accuracy.
# ### 2.)Cat Boosting (CatBoost): The second-highest performer with a score of 40.97. CatBoost is known for its robust handling of categorical features and is particularly effective for classification and regression tasks.
# ### 3.)Gradient Boosting: Also performed well with a score of 40.21. Gradient Boosting methods sequentially build weak learners to minimize errors, often yielding high predictive accuracy.
# ### 4.)Random Forest: Achieved a respectable score of 38.91. Random Forest is known for its ability to handle complex datasets with high dimensionality and provide good performance in regression tasks.
# ### 5.)Extended Gradient Boosting (XGBoost): Showed moderate performance with a score of 36.00. XGBoost is a powerful boosting algorithm known for its speed and performance optimizations.
# ### 6.)Linear Regression: Had a significantly lower score of 10.56, indicating poor predictive performance compared to the ensemble methods. Linear Regression may not capture complex nonlinear relationships present in the data.
# ### 7.)K Nearest Neighbors (KNN): Scored the lowest with a score of 1.97, suggesting inadequate performance for this regression task. KNN's effectiveness often depends on appropriate feature scaling and distance metrics.
# ### 8.)Decision Tree: Yielded a negative score of -20.64, indicating substantial underperformance. Decision Trees can overfit easily and may not generalize well to unseen data.
# 
# ## Model Evaluation and Recommendations 📝 :
# 
# ### 1.)Ensemble Methods: Light Gradient Boosting, CatBoost, and Gradient Boosting emerged as top performers, showcasing the effectiveness of boosting techniques for regression tasks.
# ### 2.)Avoidance of Simple Models: Linear Regression and K Nearest Neighbors performed poorly compared to ensemble methods, likely due to their inherent limitations in capturing complex relationships within the data.
# ### 3.)Tree-Based Models: Random Forest showed competitive performance, benefiting from ensemble learning and tree-based architecture to handle nonlinear relationships effectively.
# ### 4,)Model Selection Criteria: The choice of the most suitable model depends on various factors, including predictive performance, computational efficiency, and interpretability. Light Gradient Boosting or CatBoost could be recommended for this task based on the highest scores achieved.
# 
# ## Considerations for Future Work ✨  :
# ### 1.)Hyperparameter Tuning: Further optimization of hyperparameters for the top-performing models (e.g., learning rate, tree depth, number of estimators) could potentially enhance predictive performance.
# ### 2.)Feature Engineering: Exploring additional features or transforming existing features may improve model accuracy and robustness.
# ### 3.)Cross-Validation: Conducting cross-validation to assess model stability and generalizability across different subsets of the data.
# ### 4.)Ensemble Strategies: Leveraging ensemble techniques such as stacking or blending of multiple models to further boost predictive accuracy.
# 
# ## Final Recommendations🌈🏆  :
# ### 1.)Based on the evaluation results, the Light Gradient Boosting model (LGB) or CatBoost (CB) appears to be the most promising for predicting movie ratings in this context.
# ### 2.)It's essential to iterate on model development by refining techniques, exploring feature engineering options, and validating performance through rigorous testing to deploy an effective and reliable predictive system.

# # THANK YOU 🌠
# 
# ## Project Made By - ROMA BATHAM 👩🏻‍💻
# ## Data Science Intern 📊📈
# ## Afame Technologies✨ 
