# Data Mining Project
Konstantinos Anagnostou

## Introduction

In this project, we apply data mining techniques to analyze the Patients
Data.xlsx dataset, which contains various health-related attributes of
patients. The goal is to analyze this data and predict whether a patient
had a heart attack using machine learning algorithms.

## First Steps

Exploratory Data Analysis (EDA)

import liblaries and dataset

``` python
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

#Load our data
data = pd.read_excel("Patients Data.xlsx")

print(data.shape)
print("Data Overview:")
print(data.head())
print("\nData Info:")
data.info()

#Check for missing values
print("\nMissing  Values:")
print(data.isnull().sum())

#Seperate categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(include=['int64','float64']).columns

#Identify binary variables , 0 or 1
binary_cols = [col for col in numerical_cols if set(data[col].dropna().unique()) <= {0, 1}]
non_binary_cols = [col for col in numerical_cols if col not in binary_cols]

print(f"\nBinary Columns:{binary_cols}")
print(f"Non-Binary Numerical Columns:{non_binary_cols}")

plt.figure(figsize=(8, 5))
sns.countplot(x='HadHeartAttack',data=data)
plt.title("Distribution of Target Variable: HadHeartAttack")
plt.xlabel('HadHeartAttack')
plt.ylabel('Count')
plt.show()


# Count occurrences of HadHeartAttack per state
state_counts = data.groupby('State')['HadHeartAttack'].sum().reset_index()
state_counts.columns = ['state', 'had_heart_attack_count']

# Load US states shapefile (can use geopandas datasets or an external GeoJSON)
url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
us_states = gpd.read_file(url)

# Merge state data with our counts
us_states = us_states.merge(state_counts, left_on='name', right_on='state', how='left')
us_states['had_heart_attack_count'] = us_states['had_heart_attack_count'].fillna(0)  # Fill missing states with zero

# Plot heatmap
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
us_states.plot(column='had_heart_attack_count', cmap='Reds', linewidth=0.8, edgecolor='black', legend=True, ax=ax)
ax.set_title("Heatmap of HadHeartAttack Cases by State", fontsize=15)
ax.axis("off")

plt.show()
```

    (237630, 35)
    Data Overview:
       PatientID    State     Sex GeneralHealth   AgeCategory  HeightInMeters  \
    0          1  Alabama  Female          Fair  Age 75 to 79            1.63   
    1          2  Alabama  Female     Very good  Age 65 to 69            1.60   
    2          3  Alabama    Male     Excellent  Age 60 to 64            1.78   
    3          4  Alabama    Male     Very good  Age 70 to 74            1.78   
    4          5  Alabama  Female          Good  Age 50 to 54            1.68   

       WeightInKilograms        BMI  HadHeartAttack  HadAngina  ...  \
    0          84.820000  32.099998               0          1  ...   
    1          71.669998  27.990000               0          0  ...   
    2          71.209999  22.530001               0          0  ...   
    3          95.250000  30.129999               0          0  ...   
    4          78.019997  27.760000               0          0  ...   

                                 ECigaretteUsage  ChestScan  \
    0  Never used e-cigarettes in my entire life          1   
    1  Never used e-cigarettes in my entire life          0   
    2  Never used e-cigarettes in my entire life          0   
    3  Never used e-cigarettes in my entire life          0   
    4  Never used e-cigarettes in my entire life          1   

          RaceEthnicityCategory  AlcoholDrinkers  HIVTesting  FluVaxLast12  \
    0  White only, Non-Hispanic                0           0             0   
    1  White only, Non-Hispanic                0           0             1   
    2  White only, Non-Hispanic                1           0             0   
    3  White only, Non-Hispanic                0           0             1   
    4  Black only, Non-Hispanic                0           0             1   

       PneumoVaxEver                                  TetanusLast10Tdap  \
    0              1  No, did not receive any tetanus shot in the pa...   
    1              1                                 Yes, received Tdap   
    2              0  Yes, received tetanus shot but not sure what type   
    3              1  Yes, received tetanus shot but not sure what type   
    4              0  No, did not receive any tetanus shot in the pa...   

       HighRiskLastYear  CovidPos  
    0                 0         1  
    1                 0         0  
    2                 0         0  
    3                 0         0  
    4                 0         0  

    [5 rows x 35 columns]

    Data Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 237630 entries, 0 to 237629
    Data columns (total 35 columns):
     #   Column                     Non-Null Count   Dtype  
    ---  ------                     --------------   -----  
     0   PatientID                  237630 non-null  int64  
     1   State                      237630 non-null  object 
     2   Sex                        237630 non-null  object 
     3   GeneralHealth              237630 non-null  object 
     4   AgeCategory                237630 non-null  object 
     5   HeightInMeters             237630 non-null  float64
     6   WeightInKilograms          237630 non-null  float64
     7   BMI                        237630 non-null  float64
     8   HadHeartAttack             237630 non-null  int64  
     9   HadAngina                  237630 non-null  int64  
     10  HadStroke                  237630 non-null  int64  
     11  HadAsthma                  237630 non-null  int64  
     12  HadSkinCancer              237630 non-null  int64  
     13  HadCOPD                    237630 non-null  int64  
     14  HadDepressiveDisorder      237630 non-null  int64  
     15  HadKidneyDisease           237630 non-null  int64  
     16  HadArthritis               237630 non-null  int64  
     17  HadDiabetes                237630 non-null  object 
     18  DeafOrHardOfHearing        237630 non-null  int64  
     19  BlindOrVisionDifficulty    237630 non-null  int64  
     20  DifficultyConcentrating    237630 non-null  int64  
     21  DifficultyWalking          237630 non-null  int64  
     22  DifficultyDressingBathing  237630 non-null  int64  
     23  DifficultyErrands          237630 non-null  int64  
     24  SmokerStatus               237630 non-null  object 
     25  ECigaretteUsage            237630 non-null  object 
     26  ChestScan                  237630 non-null  int64  
     27  RaceEthnicityCategory      237630 non-null  object 
     28  AlcoholDrinkers            237630 non-null  int64  
     29  HIVTesting                 237630 non-null  int64  
     30  FluVaxLast12               237630 non-null  int64  
     31  PneumoVaxEver              237630 non-null  int64  
     32  TetanusLast10Tdap          237630 non-null  object 
     33  HighRiskLastYear           237630 non-null  int64  
     34  CovidPos                   237630 non-null  int64  
    dtypes: float64(3), int64(23), object(9)
    memory usage: 63.5+ MB

    Missing  Values:
    PatientID                    0
    State                        0
    Sex                          0
    GeneralHealth                0
    AgeCategory                  0
    HeightInMeters               0
    WeightInKilograms            0
    BMI                          0
    HadHeartAttack               0
    HadAngina                    0
    HadStroke                    0
    HadAsthma                    0
    HadSkinCancer                0
    HadCOPD                      0
    HadDepressiveDisorder        0
    HadKidneyDisease             0
    HadArthritis                 0
    HadDiabetes                  0
    DeafOrHardOfHearing          0
    BlindOrVisionDifficulty      0
    DifficultyConcentrating      0
    DifficultyWalking            0
    DifficultyDressingBathing    0
    DifficultyErrands            0
    SmokerStatus                 0
    ECigaretteUsage              0
    ChestScan                    0
    RaceEthnicityCategory        0
    AlcoholDrinkers              0
    HIVTesting                   0
    FluVaxLast12                 0
    PneumoVaxEver                0
    TetanusLast10Tdap            0
    HighRiskLastYear             0
    CovidPos                     0
    dtype: int64

    Binary Columns:['HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos']
    Non-Binary Numerical Columns:['PatientID', 'HeightInMeters', 'WeightInKilograms', 'BMI']

![](NewProject_files/figure-commonmark/cell-2-output-2.png)

![](NewProject_files/figure-commonmark/cell-2-output-3.png)

Objective: We are exploring the dataset by checking its dimensions, the
first few rows, and any missing values. Action: After loading the data,
we check for missing values to understand where data preprocessing might
be needed.

According to the hitmap , the States of America with the highest number
of recorded heart attacks are Washington , Florida , Ohio and Texas.

The target variable for classification is HadHeartAttack. In this step,
i am trying to maintain as much information as possible from the
original data.

So handling categorial variables is crucial for the main analysis.

``` python
# Binary Encoding for Sex
data['Sex'] = data['Sex'].map({'Male': 0, 'Female': 1})

# Ordinal Encoding for GeneraHealth
health_order = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
data['GeneralHealth_Ordinal'] = data['GeneralHealth'].apply(lambda x: health_order.index(x) + 1)


# Binary encoding for diabetes
data['HadDiabetes1'] = data['HadDiabetes'].map({'Yes': 1, 'No': 0, 'No, pre-diabetes or borderline diabetes': 0, 'Yes, but only during pregnancy (female)': 0})

# One-Hot Encoding for SmokerStatus
data = pd.get_dummies(data, columns=['SmokerStatus'], drop_first=True)

# Binary Encoding for ECigaretteUsage
data['ECigaretteUsage1'] = data['ECigaretteUsage'].map({'Use them some days': 1, 'Never used e-cigarettes in my entire life': 1, 'Not at all (right now)': 0, 'Use them every day': 2})

# One-Hot Encoding for RaceEthnicityCategory
data = pd.get_dummies(data, columns=['RaceEthnicityCategory'], drop_first=True)

# Binary Encoding for TetanusLast10Tdap
data['Tetanus'] = data['TetanusLast10Tdap'].map({'Yes, received Tdap': 1, 'Yes, received tetanus shot but not sure what type': 1, 'Yes, received tetanus shot, but not Tdap': 1, 'No, did not receive any tetanus shot in the past 10 years': 0})
```

``` python
# Function to convert main age into mean
def convert_age_to_mean(age_range):
    try:
        
        age_range = age_range.lower().replace("age ", "")  
        if "or older" in age_range:
            return 80
        start, end = map(int, age_range.split(" to "))
        return (start + end) / 2
    except ValueError:
        print(f" Error in proseccing : {age_range}") 
        return None


data['AgeNumeric'] = data['AgeCategory'].apply(convert_age_to_mean)

# Define state-to-region mapping
region_mapping = {
    'Northeast': ['Maine', 'New Hampshire', 'Vermont', 'Massachusetts', 'Rhode Island', 'Connecticut', 'New York', 'New Jersey', 'Pennsylvania', 'District of Columbia'],
    'South': ['Delaware', 'Maryland', 'West Virginia', 'Virginia', 'Kentucky', 'Tennessee', 'North Carolina', 'South Carolina', 'Georgia', 'Alabama', 'Mississippi', 'Arkansas', 'Louisiana', 'Florida'],
    'Midwest': ['Ohio', 'Indiana', 'Illinois', 'Michigan', 'Wisconsin', 'Minnesota', 'Iowa', 'Missouri', 'North Dakota', 'South Dakota', 'Nebraska', 'Kansas'],
    'West': ['Montana', 'Idaho', 'Wyoming', 'Colorado', 'New Mexico', 'Arizona', 'Utah', 'Nevada', 'Washington', 'Oregon', 'California', 'Alaska', 'Hawaii'],
    'Southwest': ['Texas', 'Oklahoma'],
    'Territories': ['Guam', 'Puerto Rico', 'Virgin Islands']
}



# Assign regions to states
def get_region(state):
    for region, states in region_mapping.items():
        if state in states:
            return region    
    return 'Unknown'


data['Region'] = data['State'].apply(get_region)

# Drop original state column
data.drop(columns=['State'], inplace=True)

# One-hot encode the Region column
data = pd.get_dummies(data, columns=['Region'], drop_first=True)

# Display transformed data
print(data.head())

data = data.applymap(lambda x: 1 if x is True else (0 if x is False else x))

 
print(data)
```

       PatientID  Sex GeneralHealth   AgeCategory  HeightInMeters  \
    0          1    1          Fair  Age 75 to 79            1.63   
    1          2    1     Very good  Age 65 to 69            1.60   
    2          3    0     Excellent  Age 60 to 64            1.78   
    3          4    0     Very good  Age 70 to 74            1.78   
    4          5    1          Good  Age 50 to 54            1.68   

       WeightInKilograms        BMI  HadHeartAttack  HadAngina  HadStroke  ...  \
    0          84.820000  32.099998               0          1          0  ...   
    1          71.669998  27.990000               0          0          0  ...   
    2          71.209999  22.530001               0          0          0  ...   
    3          95.250000  30.129999               0          0          0  ...   
    4          78.019997  27.760000               0          0          0  ...   

       RaceEthnicityCategory_Multiracial, Non-Hispanic  \
    0                                            False   
    1                                            False   
    2                                            False   
    3                                            False   
    4                                            False   

       RaceEthnicityCategory_Other race only, Non-Hispanic  \
    0                                              False     
    1                                              False     
    2                                              False     
    3                                              False     
    4                                              False     

       RaceEthnicityCategory_White only, Non-Hispanic  Tetanus  AgeNumeric  \
    0                                            True        0        77.0   
    1                                            True        1        67.0   
    2                                            True        1        62.0   
    3                                            True        1        72.0   
    4                                           False        0        52.0   

       Region_Northeast Region_South  Region_Southwest  Region_Territories  \
    0             False         True             False               False   
    1             False         True             False               False   
    2             False         True             False               False   
    3             False         True             False               False   
    4             False         True             False               False   

       Region_West  
    0        False  
    1        False  
    2        False  
    3        False  
    4        False  

    [5 rows x 49 columns]

    C:\Users\mixaa\AppData\Local\Temp\ipykernel_1796\221098070.py:48: FutureWarning:

    DataFrame.applymap has been deprecated. Use DataFrame.map instead.

            PatientID  Sex GeneralHealth   AgeCategory  HeightInMeters  \
    0               1    1          Fair  Age 75 to 79            1.63   
    1               2    1     Very good  Age 65 to 69            1.60   
    2               3    0     Excellent  Age 60 to 64            1.78   
    3               4    0     Very good  Age 70 to 74            1.78   
    4               5    1          Good  Age 50 to 54            1.68   
    ...           ...  ...           ...           ...             ...   
    237625     237626    1          Good  Age 60 to 64            1.57   
    237626     237627    1          Good  Age 55 to 59            1.70   
    237627     237628    0          Fair  Age 45 to 49            1.75   
    237628     237629    1     Very good  Age 25 to 29            1.57   
    237629     237630    1          Good  Age 30 to 34            1.60   

            WeightInKilograms        BMI  HadHeartAttack  HadAngina  HadStroke  \
    0               84.820000  32.099998               0          1          0   
    1               71.669998  27.990000               0          0          0   
    2               71.209999  22.530001               0          0          0   
    3               95.250000  30.129999               0          0          0   
    4               78.019997  27.760000               0          0          0   
    ...                   ...        ...             ...        ...        ...   
    237625          90.720001  36.580002               0          0          0   
    237626          72.570000  25.059999               0          1          0   
    237627          70.309998  22.889999               1          1          0   
    237628          46.720001  18.840000               0          0          0   
    237629          83.010002  32.419998               0          0          0   

            ...  RaceEthnicityCategory_Multiracial, Non-Hispanic  \
    0       ...                                                0   
    1       ...                                                0   
    2       ...                                                0   
    3       ...                                                0   
    4       ...                                                0   
    ...     ...                                              ...   
    237625  ...                                                0   
    237626  ...                                                0   
    237627  ...                                                0   
    237628  ...                                                0   
    237629  ...                                                0   

            RaceEthnicityCategory_Other race only, Non-Hispanic  \
    0                                                       0     
    1                                                       0     
    2                                                       0     
    3                                                       0     
    4                                                       0     
    ...                                                   ...     
    237625                                                  0     
    237626                                                  0     
    237627                                                  0     
    237628                                                  0     
    237629                                                  0     

            RaceEthnicityCategory_White only, Non-Hispanic  Tetanus  AgeNumeric  \
    0                                                    1        0        77.0   
    1                                                    1        1        67.0   
    2                                                    1        1        62.0   
    3                                                    1        1        72.0   
    4                                                    0        0        52.0   
    ...                                                ...      ...         ...   
    237625                                               0        0        62.0   
    237626                                               0        1        57.0   
    237627                                               0        1        47.0   
    237628                                               0        0        27.0   
    237629                                               0        0        32.0   

            Region_Northeast Region_South  Region_Southwest  Region_Territories  \
    0                      0            1                 0                   0   
    1                      0            1                 0                   0   
    2                      0            1                 0                   0   
    3                      0            1                 0                   0   
    4                      0            1                 0                   0   
    ...                  ...          ...               ...                 ...   
    237625                 0            0                 0                   1   
    237626                 0            0                 0                   1   
    237627                 0            0                 0                   1   
    237628                 0            0                 0                   1   
    237629                 0            0                 0                   1   

            Region_West  
    0                 0  
    1                 0  
    2                 0  
    3                 0  
    4                 0  
    ...             ...  
    237625            0  
    237626            0  
    237627            0  
    237628            0  
    237629            0  

    [237630 rows x 49 columns]

analysis between target and the continuous variables

``` python
print(data.columns)
columns_to_drop = ['GeneralHealth', 'AgeCategory','TetanusLast10Tdap','PatientID','HadDiabetes','ECigaretteUsage']  # Droping 
data = data.drop(columns=columns_to_drop)

print(data)
data.info()
```

    Index(['PatientID', 'Sex', 'GeneralHealth', 'AgeCategory', 'HeightInMeters',
           'WeightInKilograms', 'BMI', 'HadHeartAttack', 'HadAngina', 'HadStroke',
           'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
           'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
           'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
           'DifficultyConcentrating', 'DifficultyWalking',
           'DifficultyDressingBathing', 'DifficultyErrands', 'ECigaretteUsage',
           'ChestScan', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12',
           'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos',
           'GeneralHealth_Ordinal', 'HadDiabetes1',
           'SmokerStatus_Current smoker - now smokes some days',
           'SmokerStatus_Former smoker', 'SmokerStatus_Never smoked',
           'ECigaretteUsage1', 'RaceEthnicityCategory_Hispanic',
           'RaceEthnicityCategory_Multiracial, Non-Hispanic',
           'RaceEthnicityCategory_Other race only, Non-Hispanic',
           'RaceEthnicityCategory_White only, Non-Hispanic', 'Tetanus',
           'AgeNumeric', 'Region_Northeast', 'Region_South', 'Region_Southwest',
           'Region_Territories', 'Region_West'],
          dtype='object')
            Sex  HeightInMeters  WeightInKilograms        BMI  HadHeartAttack  \
    0         1            1.63          84.820000  32.099998               0   
    1         1            1.60          71.669998  27.990000               0   
    2         0            1.78          71.209999  22.530001               0   
    3         0            1.78          95.250000  30.129999               0   
    4         1            1.68          78.019997  27.760000               0   
    ...     ...             ...                ...        ...             ...   
    237625    1            1.57          90.720001  36.580002               0   
    237626    1            1.70          72.570000  25.059999               0   
    237627    0            1.75          70.309998  22.889999               1   
    237628    1            1.57          46.720001  18.840000               0   
    237629    1            1.60          83.010002  32.419998               0   

            HadAngina  HadStroke  HadAsthma  HadSkinCancer  HadCOPD  ...  \
    0               1          0          1              1        0  ...   
    1               0          0          0              0        0  ...   
    2               0          0          0              0        0  ...   
    3               0          0          0              0        0  ...   
    4               0          0          0              0        0  ...   
    ...           ...        ...        ...            ...      ...  ...   
    237625          0          0          0              0        0  ...   
    237626          1          0          0              0        0  ...   
    237627          1          0          0              0        0  ...   
    237628          0          0          0              0        0  ...   
    237629          0          0          0              0        0  ...   

            RaceEthnicityCategory_Multiracial, Non-Hispanic  \
    0                                                     0   
    1                                                     0   
    2                                                     0   
    3                                                     0   
    4                                                     0   
    ...                                                 ...   
    237625                                                0   
    237626                                                0   
    237627                                                0   
    237628                                                0   
    237629                                                0   

            RaceEthnicityCategory_Other race only, Non-Hispanic  \
    0                                                       0     
    1                                                       0     
    2                                                       0     
    3                                                       0     
    4                                                       0     
    ...                                                   ...     
    237625                                                  0     
    237626                                                  0     
    237627                                                  0     
    237628                                                  0     
    237629                                                  0     

            RaceEthnicityCategory_White only, Non-Hispanic  Tetanus  AgeNumeric  \
    0                                                    1        0        77.0   
    1                                                    1        1        67.0   
    2                                                    1        1        62.0   
    3                                                    1        1        72.0   
    4                                                    0        0        52.0   
    ...                                                ...      ...         ...   
    237625                                               0        0        62.0   
    237626                                               0        1        57.0   
    237627                                               0        1        47.0   
    237628                                               0        0        27.0   
    237629                                               0        0        32.0   

            Region_Northeast  Region_South  Region_Southwest  Region_Territories  \
    0                      0             1                 0                   0   
    1                      0             1                 0                   0   
    2                      0             1                 0                   0   
    3                      0             1                 0                   0   
    4                      0             1                 0                   0   
    ...                  ...           ...               ...                 ...   
    237625                 0             0                 0                   1   
    237626                 0             0                 0                   1   
    237627                 0             0                 0                   1   
    237628                 0             0                 0                   1   
    237629                 0             0                 0                   1   

            Region_West  
    0                 0  
    1                 0  
    2                 0  
    3                 0  
    4                 0  
    ...             ...  
    237625            0  
    237626            0  
    237627            0  
    237628            0  
    237629            0  

    [237630 rows x 43 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 237630 entries, 0 to 237629
    Data columns (total 43 columns):
     #   Column                                               Non-Null Count   Dtype  
    ---  ------                                               --------------   -----  
     0   Sex                                                  237630 non-null  int64  
     1   HeightInMeters                                       237630 non-null  float64
     2   WeightInKilograms                                    237630 non-null  float64
     3   BMI                                                  237630 non-null  float64
     4   HadHeartAttack                                       237630 non-null  int64  
     5   HadAngina                                            237630 non-null  int64  
     6   HadStroke                                            237630 non-null  int64  
     7   HadAsthma                                            237630 non-null  int64  
     8   HadSkinCancer                                        237630 non-null  int64  
     9   HadCOPD                                              237630 non-null  int64  
     10  HadDepressiveDisorder                                237630 non-null  int64  
     11  HadKidneyDisease                                     237630 non-null  int64  
     12  HadArthritis                                         237630 non-null  int64  
     13  DeafOrHardOfHearing                                  237630 non-null  int64  
     14  BlindOrVisionDifficulty                              237630 non-null  int64  
     15  DifficultyConcentrating                              237630 non-null  int64  
     16  DifficultyWalking                                    237630 non-null  int64  
     17  DifficultyDressingBathing                            237630 non-null  int64  
     18  DifficultyErrands                                    237630 non-null  int64  
     19  ChestScan                                            237630 non-null  int64  
     20  AlcoholDrinkers                                      237630 non-null  int64  
     21  HIVTesting                                           237630 non-null  int64  
     22  FluVaxLast12                                         237630 non-null  int64  
     23  PneumoVaxEver                                        237630 non-null  int64  
     24  HighRiskLastYear                                     237630 non-null  int64  
     25  CovidPos                                             237630 non-null  int64  
     26  GeneralHealth_Ordinal                                237630 non-null  int64  
     27  HadDiabetes1                                         237630 non-null  int64  
     28  SmokerStatus_Current smoker - now smokes some days   237630 non-null  int64  
     29  SmokerStatus_Former smoker                           237630 non-null  int64  
     30  SmokerStatus_Never smoked                            237630 non-null  int64  
     31  ECigaretteUsage1                                     237630 non-null  int64  
     32  RaceEthnicityCategory_Hispanic                       237630 non-null  int64  
     33  RaceEthnicityCategory_Multiracial, Non-Hispanic      237630 non-null  int64  
     34  RaceEthnicityCategory_Other race only, Non-Hispanic  237630 non-null  int64  
     35  RaceEthnicityCategory_White only, Non-Hispanic       237630 non-null  int64  
     36  Tetanus                                              237630 non-null  int64  
     37  AgeNumeric                                           237630 non-null  float64
     38  Region_Northeast                                     237630 non-null  int64  
     39  Region_South                                         237630 non-null  int64  
     40  Region_Southwest                                     237630 non-null  int64  
     41  Region_Territories                                   237630 non-null  int64  
     42  Region_West                                          237630 non-null  int64  
    dtypes: float64(4), int64(39)
    memory usage: 78.0 MB

## Classification 1

# KNN Algorithmn

K-Nearest Neighbors (KNN) is a simple and intuitive machine learning
algorithm used for classification and regression tasks. It is a
non-parametric, lazy learning algorithm that relies on the similarity
between data points to make predictions.

When a new data point needs to be classified, KNN calculates the
distance between the query point and all points in the training dataset
using a distance metric such as Euclidean distance. It then identifies
the 𝑘 k-closest points, known as the nearest neighbors.

For classification, the algorithm assigns the class that is most common
among the 𝑘 k neighbors (majority voting).

## Dimentionality Reduction

Principal Component Analysis (PCA) is a widely used technique for
dimensionality reduction in machine learning. It transforms a dataset
with many features into a smaller set of uncorrelated variables called
principal components, while retaining as much variance in the data as
possible.

PCA works by:

1.Standardizing the Data: Ensuring all features have zero mean and unit
variance.

2.Computing the Covariance Matrix: To capture relationships between
features.

3.Finding Eigenvectors and Eigenvalues: The eigenvectors represent the
directions (principal components), and the eigenvalues indicate the
amount of variance captured by each component.

4.Selecting Principal Components: The components with the largest
eigenvalues are chosen, as they explain the most variance in the data.

5.Transforming the Data: The original data is projected onto the
selected principal components, reducing its dimensionality.

``` python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Drop target column and separate features
y = data['HadHeartAttack']
X = data.drop(columns=['HadHeartAttack'])

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Compute explained variance for each number of components
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid()
plt.show()

# Apply PCA and reduce to 3 components for visualization
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)

# Visualization in three dimensions using the first 3 principal components
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y, cmap='viridis', alpha=0.7)
ax.set_title('PCA Visualization (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
legend = ax.legend(*scatter.legend_elements(), title="Target")
ax.add_artist(legend)
plt.show()

# Apply PCA and reduce to 20 components for KNN
pca_20 = PCA(n_components=20)
X_pca_20 = pca_20.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_20, y, test_size=0.2, random_state=42)

# Train and predict with KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate Accuracy and F1 Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix for Sensitivity calculation
cm = confusion_matrix(y_test, y_pred)

# Sensitivity (True Positive Rate)
sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # TP / (TP + FN)

# Printing the results
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")

# Optionally, you can plot the confusion matrix for better visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Attack', 'Heart Attack'], yticklabels=['No Heart Attack', 'Heart Attack'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

![](NewProject_files/figure-commonmark/cell-6-output-1.png)

![](NewProject_files/figure-commonmark/cell-6-output-2.png)

    Accuracy: 0.9330
    F1 Score: 0.1190
    Sensitivity: 0.0811

![](NewProject_files/figure-commonmark/cell-6-output-4.png)

## Classification 2

# Logistic Regression

1.  **Linear Relationship**: Logistic regression starts by modeling a
    linear relationship between the features and the log-odds of the
    outcome: $$
    z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
    $$ Here, ( z ) is the linear combination of the input features ( x_i
    ), their weights ( w_i ), and the bias term ( b ).

2.  **Logistic Function**: To map ( z ) (which can range from ( -) to (
    +)) to a probability between 0 and 1, it uses the **sigmoid
    function**: $$
    \sigma(z) = \frac{1}{1 + e^{-z}}
    $$ The output of the sigmoid function represents the probability of
    the positive class.

3.  **Prediction**:

    - If ( (z) ), the model predicts the positive class (( 1 )).
    - If ( (z) \< 0.5 ), it predicts the negative class (( 0 )).

4.  **Training the Model**:

    - Logistic regression learns the weights ( w ) and bias ( b ) by
      minimizing a **log-loss** (or binary cross-entropy) cost function.
    - Optimization algorithms like **Gradient Descent** are used to find
      the best parameters.

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

# 1. Data preprocessing
X = data.drop(columns=['HadHeartAttack'])
y = data['HadHeartAttack']

# Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 2. Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# 3. Logistic Regression training
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 4. Predictions and probabilities
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Calculate Accuracy and AUC (Area Under Curve)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy (Logistic Regression): {accuracy:.4f}")
print(f"ROC AUC (Logistic Regression): {roc_auc:.4f}")

# 5. Display the feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance (Top Features):")
print(feature_importance.head())

# 6. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.4f})', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

    Accuracy (Logistic Regression): 0.9456
    ROC AUC (Logistic Regression): 0.8787
    Feature Importance (Top Features):
             Feature  Coefficient
    4      HadAngina     2.447478
    36    AgeNumeric     2.156551
    5      HadStroke     0.843378
    18     ChestScan     0.606353
    26  HadDiabetes1     0.377558

![](NewProject_files/figure-commonmark/cell-7-output-2.png)

## Conclusion

This project demonstrates the process of applying data mining techniques
to predict heart disease. Through encoding, visualization,
dimensionality reduction, and classification, we build models that
predict whether a patient had a heart attack

# References

1.  Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern
    classification. *IEEE Transactions on Information Theory, 13*(1),
    21-27.
2.  Jolliffe, I. T. (2002). *Principal component analysis*. Springer
    Series in Statistics.
3.  Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied
    Logistic Regression*. Wiley.
