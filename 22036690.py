import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(file_path):
    
    
    """
    Read data from a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data.
    """
    return pd.read_csv(file_path)


def fill_missing_values_with_mean(df, column_name):
    
    
    """
    Fill missing values in a DataFrame column with the mean values based on 'country'.

    Args:
            df (pd.DataFrame): Input DataFrame.
        column_name (str): Name of the column with missing values to be filled.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    mean_by_country = df.groupby('country')[column_name].transform('mean')
    df[column_name] = df[column_name].fillna(mean_by_country)
    return df


def extract_year_from_date(df):
    
    
    """
    Extract the 'year' from the 'date' column and add it as a new column.
    
    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the 'year' column added.
    """
   
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    return df


def create_line_plot(data_frame, top_n = 5):
    
    
    """
    Create and display a line plot for the top N countries with the highest mean inflation.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        top_n (int): Number of top countries to include in the plot.
    """
    # Your code for creating a line plot here
    top_countries = data_frame.groupby('country')['Inflation'].mean().nlargest(top_n)
    plt.figure(figsize = (12, 6))
    for country in top_countries.index:
        country_data = data_frame[data_frame['country'] == country].groupby('year')['Inflation'].mean()
        plt.plot(country_data.index, country_data.values, linestyle = '-', label = country)
    plt.xlabel('Year')
    plt.ylabel('Mean Inflation')
    plt.title('Yearly Mean Inflation by Top {} Countries'.format(top_n))
    plt.grid(True)
    plt.legend(title = 'Country', bbox_to_anchor = (1, 1))
    plt.tight_layout()
    plt.show()


def create_bar_chart(data_frame, top_n = 5):
    
    
    """
    Create and display a bar chart for the top N countries with the highest mean inflation.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        top_n (int): Number of top countries to include in the plot.
    """
        # Calculate the mean "Open," "High," "Low," "Close," and "Inflation" for each country
    #print(df)
    mean_values_by_country = df.groupby('country')[['Open', 'High', 'Low', 'Close', 'Inflation']].mean()

    # Sort the countries based on mean inflation in descending order and select the top 5
    top_countries = mean_values_by_country['Inflation'].sort_values(ascending = False).head(5)
    print(top_countries)
    plt.figure(figsize = (12, 6))
    #print(top_countries.index)
    for country in top_countries.index:
        # Filter the DataFrame for the current country
        country_data = df[df['country'] == country]
        #print(country_data)
        # Extract the relevant columns
        country_data = mean_values_by_country.loc[country]
        indicators = country_data.index
        #print(country_data)
        values = country_data.values

        # Create bar plots for each column
        plt.bar(indicators, values, label = country)

    # Set labels, title, and legend
    plt.xlabel('Indicators')
    plt.ylabel('Mean Values')
    plt.title('Bar Chart of Top 5 Countries (Highest Mean Inflation')
    plt.legend()
    plt.xticks(rotation = 45)
    plt.show()
  

def create_box_plot(data_frame, top_n = 5):
    
    
    """
    Create and display a box plot for the top N countries with the highest mean inflation.

    Args:
        data_frame (pd.DataFrame): Data for plotting.
        top_n (int): Number of top countries to include in the plot.
    """
    # Your code for creating a box plot here
    top_countries = data_frame.groupby('country')['Inflation'].mean().nlargest(top_n)
    top_countries_data = data_frame[data_frame['country'].isin(top_countries.index)]
    plt.figure(figsize = (10, 6))
    plt.boxplot([top_countries_data[top_countries_data['country'] == country]['Inflation'] for country in top_countries.index], labels = top_countries.index)
    plt.xlabel('Countries')
    plt.ylabel('Mean Inflation')
    plt.title('Box Plot of Mean Inflation for Top {} Countries (2007-2023)'.format(top_n))
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    
    file_path = "WLD_RTFP_country_2023-10-02.csv"
    df = read_data(file_path)

    # Fill missing values in 'Inflation' column with mean values by country
    df = fill_missing_values_with_mean(df, 'Inflation')
    
    # Extract the 'year' from the 'date' column
    df = extract_year_from_date(df)
   
    
    # Line plot for top 5 countries with highest mean inflation
    create_line_plot(df, top_n = 5)


    # Bar chart for top 5 countries with highest mean inflation
    create_bar_chart(df, top_n = 5)


    # Box plot for top 5 countries with highest mean inflation
    create_box_plot(df, top_n = 5)
    
    
    # Filter the dataset for the specified countries
    countries = ['Sudan', 'South Sudan', 'Lebanon', 'Syrian Arab Republic', 'Haiti']
    filtered_data = df[df['country'].isin(countries)]
    
    # Group the data by country to perform the calculations for each country
    grouped_data = filtered_data.groupby('country')
    
    for country, df in grouped_data:
        
        
        if(country == "Sudan"):
            
            print(f"Statistics for {country}:\n")
            
            # Basic descriptive statistics
            desc_stats = df.describe()
            print("Descriptive Statistics:\n", desc_stats, "\n")
            
            # Calculating mean, median, standard deviation, skewness, kurtosis
            numeric_df = df.select_dtypes(include=[float, int])  # Filter only numeric columns
            mean = numeric_df.mean()
            median = numeric_df.median()
            std_dev = numeric_df.std()
            skewness = numeric_df.skew()
            kurtosis = numeric_df.kurtosis()
            print("Mean:\n", mean, "\n")
            print("Median:\n", median, "\n")
            print("Standard Deviation:\n", std_dev, "\n")
            print("Skewness:\n", skewness, "\n")
            print("Kurtosis:\n", kurtosis, "\n")
            
            # Generating a correlation matrix for the numerical columns
            corr_matrix = numeric_df.corr()
            print("Correlation Matrix:\n", corr_matrix, "\n\n")
   