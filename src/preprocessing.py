import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import LabelEncoder
def load_and_process_data():
    df=pd.read_csv("data/raw/retail_customers_COMPLETE_CATEGORICAL.csv")
        
    print("---------------- exploring the data... ----------------\n")
    print(df.head())
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    print("number of rows is :",len(df.index))
    return df

def fix_age_column(df):
    print("----------------fixing age column...------------------\n")
    print(df["Age"].isnull().value_counts())

    skewness = df['Age'].skew()
    print("Skewness of Age:", skewness)
    #skew to see the symetry of the age values 

    #it's almost symetric => applying mean 
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    print("creating a new dataframe with filled age column...")
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    print(df["Age"].isnull().value_counts())
    print("column age is now filled with mean values")
    return df

def fix_support_tickets_and_satisfaction(df):
    print("----------------SupportTickets, Satisfaction------------------\n")


    print("min SupportTicketsCount:", min(df["SupportTicketsCount"]))
    print("max SupportTicketsCount:", max(df["SupportTicketsCount"]))
    print("value counts SupportTicketsCount:\n", df["SupportTicketsCount"].value_counts())
    skewness = df['SupportTicketsCount'].skew()
    print("Skewness of SupportTicketsCount", skewness)

    print("min SatisfactionScore:", min(df["SatisfactionScore"]))
    print("max SatisfactionScore:", max(df["SatisfactionScore"]))
    print("value counts SatisfactionScore:\n", df["SatisfactionScore"].value_counts())
    skewness = df['SatisfactionScore'].skew()
    print("Skewness of SatisfactionScore", skewness)


    # SupportTicketsCount: median
    print("\nfilling with the central value of the range [0,7]\n")
    median_support = df.loc[df["SupportTicketsCount"].between(0,7), "SupportTicketsCount"].median()
    df.loc[~df["SupportTicketsCount"].between(0,7), "SupportTicketsCount"] = median_support
    print("value counts SupportTicketsCount:\n", df["SupportTicketsCount"].value_counts())
    skewness = df['SupportTicketsCount'].skew()
    print("Skewness of SupportTicketsCount after median", skewness)

    # SatisfactionScore: mode
    print("\nfilling with the most frequent value in the range [0,5]\n")
    mode_satisfaction = df.loc[df["SatisfactionScore"].between(0,5), "SatisfactionScore"].mode()[0]
    df.loc[~df["SatisfactionScore"].between(0,5), "SatisfactionScore"] = mode_satisfaction
    print("value counts SatisfactionScore:\n", df["SatisfactionScore"].value_counts())
    skewness = df['SatisfactionScore'].skew()
    print("Skewness of SatisfactionScore after mode", skewness)
    return df

def parse_date_robust(date_str):
    """Parse dates in multiple formats. Returns pd.Timestamp or pd.NaT."""
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # Format: YYYY-MM-DD (ISO format)
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except:
        pass
    
    # Format: DD/MM/YYYY or DD/MM/YY (European)
    try:
        return pd.to_datetime(date_str, format='%d/%m/%Y')
    except:
        try:
            return pd.to_datetime(date_str, format='%d/%m/%y')
        except:
            pass
    
    # Format: MM/DD/YYYY (US format)
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except:
        pass
    
    # Format: MM/DD/YY (US short format)
    try:
        return pd.to_datetime(date_str, format='%m/%d/%y')
    except:
        pass
    
    # Last resort: let pandas infer
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return pd.NaT

def fix_AvgDaysBetweenPurchases(df):
    df.drop("AvgDaysBetweenPurchases", axis=1, inplace=True)
    return df

def fix_registration_date(df):
    print("----------------registration date------------------")
    print("Sample values:", df["RegistrationDate"].head(10).tolist())
    
    # Count missing before
    missing_before = df["RegistrationDate"].isna().sum()
    print(f"Missing values before parsing: {missing_before}")
    
    # Apply robust parsing
    df["RegistrationDate"] = df["RegistrationDate"].apply(parse_date_robust)
    
    # Count missing after parsing
    missing_after = df["RegistrationDate"].isna().sum()
    parsed_count = len(df) - missing_after
    print(f"Successfully parsed: {parsed_count} / {len(df)}")
    print(f"Still missing: {missing_after}")
    
    # Extract components (will be NaN for NaT dates)
    df["RegistrationDate_day"] = df["RegistrationDate"].dt.day
    df["RegistrationDate_month"] = df["RegistrationDate"].dt.month
    df["RegistrationDate_year"] = df["RegistrationDate"].dt.year
    
    # Fill missing dates with median values
    if missing_after > 0:
        median_day = int(df["RegistrationDate_day"].median())
        median_month = int(df["RegistrationDate_month"].median())
        median_year = int(df["RegistrationDate_year"].median())
        
        df["RegistrationDate_day"] = df["RegistrationDate_day"].fillna(median_day)
        df["RegistrationDate_month"] = df["RegistrationDate_month"].fillna(median_month)
        df["RegistrationDate_year"] = df["RegistrationDate_year"].fillna(median_year)
        
        print(f"Filled missing with median: day={median_day}, month={median_month}, year={median_year}")
    
    df.drop(columns=["RegistrationDate"], inplace=True)
    return df

def fix_newsletter_subscribed(df):
    print("----------------NewsletterSubscribed------------------")
    df.drop(columns=["NewsletterSubscribed"], inplace=True)
    print(df.describe())
    print(len(df.columns))
    return df

def extract_ip_features(df, ip_col='LastLoginIP'):
    print("----------------Extracting IP Features------------------")
    """
    Extrait uniquement IsPrivateIP et IsSharedSubnet depuis LastLoginIP.
    """
    df = df.copy()
    
    # 1. IsPrivateIP
    df['IsPrivateIP'] = df[ip_col].apply(
        lambda x: ipaddress.ip_address(x).is_private if pd.notna(x) else False
    )
    print("---------IsPrivateIP---------")
    print(df['IsPrivateIP'].value_counts())
    df.drop(columns=["LastLoginIP"], inplace=True) 
    print("IsPrivateIP vs Churn:")
    print(pd.crosstab(df["IsPrivateIP"], df["Churn"], normalize='index'))
       
    return df

def fix_account_status_and_churn(df):
    print("----------------AccountStatus, Churn------------------")
    print("unique values:")
    print(len(df["AccountStatus"].unique()))
    print("possible values of AccountStatus:", sorted(df["AccountStatus"].unique()))
    print("frequencies of AccountStatus:")
    print(df["AccountStatus"].value_counts())
    print("AccountStatus vs Churn:")
    print(pd.crosstab(df["AccountStatus"], df["Churn"], normalize='index'))
    df['AccountStatus'] = df['AccountStatus'].replace(
        ['Suspended', 'Pending', 'Closed'], 'Inactive'
    )
    print("AccountStatus after replacement:")
    print(df["AccountStatus"].value_counts())
    return df



def encoding(df):
    print("------------------------Encoding------------------------")
    # Include Unknown in mapping (-1 = unknown/missing)
    df = pd.get_dummies(df, columns=["Gender"], prefix="Gender", dummy_na=False)
    print(df["Gender_F"].value_counts())
    print("one hot encoding on Gender completed")

    df = pd.get_dummies(df, columns=["AccountStatus"], prefix="AccountStatus", dummy_na=False)
    print(df["AccountStatus_Active"].value_counts())
    print("one hot encoding on AccountStatus completed")
    
    print("-----------Region Encoding-----------")
    mapping = {'UK': 0, 'Europe continentale': 1, 'Océanie': 2, 'Europe du Nord': 3, 'Autre': 4,
    'Europe centrale': 5, 'Europe du Sud': 5, "Europe de l'Est": 5, 'Asie': 4, 'Moyen-Orient': 4,
    'Amérique du Nord': 6, 'Amérique du Sud': 6, 'Afrique': 4}
    # Application du mapping (Unknown deviendra NaN par défaut)
    df['Region_encoded'] = df['Region'].map(mapping)
    df=pd.get_dummies(df, columns=["Region_encoded"], prefix="Region", dummy_na=False)
    print("Region encoding completed")
    df.drop(columns=["Region"], inplace=True)

    print("-----------Favorite Season Encoding-----------")
    print(df["FavoriteSeason"].value_counts())
    df = pd.get_dummies(df, columns=["FavoriteSeason"], prefix="Season", dummy_na=False)
    print("Favorite Season encoding completed")
    

    print("-----------Preferred Time of Day Encoding-----------")
    print("PreferredTimeOfDay value counts:",df["PreferredTimeOfDay"].value_counts())
    df = pd.get_dummies(df, columns=["PreferredTimeOfDay"], prefix="TimeOfDay", dummy_na=False)
    print("Preferred Time of Day encoding completed")
    

    print("------------WeekendPreference-----------------------")
    print("WeekendPreference value counts:",df["WeekendPreference"].value_counts())
    df = pd.get_dummies(df, columns=["WeekendPreference"], prefix="Weekend", dummy_na=False)
    print("WeekendPreference encoding completed")
    

    print("-----------Customer Type Encoding-----------")
    print("CustomerType value counts:",df["CustomerType"].value_counts())
    df = pd.get_dummies(df, columns=["CustomerType"], prefix="CustomerType", dummy_na=False)
    print("Customer Type encoding completed")
    

    print("--------------LABEL ENCODING (ordered features)--------------")
    label_cols = [
        "LoyaltyLevel",
        "AgeCategory",
        "SpendingCategory",
        "ChurnRiskCategory",
        "BasketSizeCategory",
        "ProductDiversity",
        "RFMSegment"
    ]
    for col in label_cols:
        print(f"{col} value counts:",df[col].value_counts())

    print("--------------MANUAL ENCODING (ordered features)--------------")

    # 1. LoyaltyLevel: Customer lifecycle order
    loyalty_mapping = {
        'Nouveau': 0,      # New customer
        'Jeune': 1,        # Young relationship
        'Établi': 2,       # Established
        'Ancien': 3        # Old/Long-term
    }
    df['LoyaltyLevel'] = df['LoyaltyLevel'].map(loyalty_mapping)
    print("LoyaltyLevel encoding completed")

    # 2. AgeCategory: Age order (Inconnu first, then ascending)
    age_mapping = {
        'Inconnu': 0,
        '18-24': 1,
        '25-34': 2,
        '35-44': 3,
        '45-54': 4,
        '55-64': 5,
        '65+': 6
    }
    df['AgeCategory'] = df['AgeCategory'].map(age_mapping)
    print("AgeCategory encoding completed")

    # 3. SpendingCategory: Low to High
    spending_mapping = {
        'Low': 0,
        'Medium': 1,
        'High': 2,
        'VIP': 3
    }
    df['SpendingCategory'] = df['SpendingCategory'].map(spending_mapping)

    # 4. ChurnRiskCategory: Risk level (Faible=Low to Critique=Critical)
    churn_risk_mapping = {
        'Faible': 0,
        'Moyen': 1,
        'Élevé': 2,
        'Critique': 3
    }
    df['ChurnRiskCategory'] = df['ChurnRiskCategory'].map(churn_risk_mapping)
    print("ChurnRiskCategory encoding completed")

    # 5. BasketSizeCategory: Size order (Petit=Small to Grand=Large)
    basket_mapping = {
        'Petit': 0,
        'Moyen': 1,
        'Grand': 2
    }
    df['BasketSizeCategory'] = df['BasketSizeCategory'].map(basket_mapping)
    print("BasketSizeCategory encoding completed")

    # 6. ProductDiversity: Specialization level (Spécialisé=Focused to Explorateur=Explorer)
    diversity_mapping = {
        'Spécialisé': 0,   # Buys few product types
        'Modéré': 1,       # Moderate variety
        'Explorateur': 2   # Buys many different products
    }
    df['ProductDiversity'] = df['ProductDiversity'].map(diversity_mapping)
    print("ProductDiversity encoding completed")

    # 7. RFMSegment: Customer value segments (Dormants=Low to Champions=High)
    rfm_mapping = {
        'Dormants': 0,     # Inactive
        'Potentiels': 1,   # Potential
        'Fidèles': 2,      # Loyal
        'Champions': 3     # Best customers
    }
    df['RFMSegment'] = df['RFMSegment'].map(rfm_mapping)

    # 8. Country
    churn_risk_mapping = {
    'United Kingdom':1,
    'France':2,
    'Australia':3,
    'Netherlands':4,
    'Germany':5,
    'Norway':6,
    'EIRE':7,
    'Switzerland':8,
    'Spain':9,
    'Poland':10,
    'Portugal':11,
    'Italy':12,
    'Belgium':13,
    'Lithuania':14,
    'Japan':15,
    'Iceland':16,
    'Channel Islands':17,
    'Denmark':18,
    'Cyprus':19,
    'Sweden':20,
    'Austria':21,
    'Israel':22,
    'Finland':23,
    'Greece':24,
    'Singapore':25,
    'Lebanon':26,
    'United Arab Emirates':27,
    'Saudi Arabia':28,
    'Czech Republic':29,
    'Canada':30,
    'Unspecified':40,
    'Brazil':41,
    'USA':42,
    'European Community':43,
    'Bahrain':44,
    'Malta':45,
    'RSA':46
    }
    df['Country'] = df['Country'].map(churn_risk_mapping)
    print("Country encoding completed")

    print("✅encoding completed!")
    print("Checking for any unmapped (NaN) values...")
    for col in ['LoyaltyLevel', 'AgeCategory', 'SpendingCategory', 
                'ChurnRiskCategory', 'BasketSizeCategory', 'ProductDiversity', 'RFMSegment']:
        if df[col].isnull().sum() > 0:
            print(f"WARNING: {col} has {df[col].isnull().sum()} unmapped values!")
        else:
            print(f"✅ {col}: OK")
        
    print("--------------testing dataset--------------")
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum().sum())

    # print("Features:", X.shape)
    # print("Target:", y.shape)
    # print(X.isnull().sum().sum())
    # print(y.value_counts(normalize=True))
    return df

def feature_engineering(df):
    print("----------------feature engineering------------------")
    # Drop perfect multicollinearity beacuse correlation is almost 1
    df.drop(columns=['UniqueDescriptions', 'UniqueInvoices', 'AvgLinesPerInvoice'], inplace=True)

    # 1. Daily spending velocity (monetary efficiency)
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # 2. Average basket value (order value)
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']

    # 3. Recency vs Tenure ratio (customer activity trend)
    # Values close to 0 = very active; close to 1 = inactive despite long history
    df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)

    # 4. Product diversity per transaction (variety buyer vs bulk buyer)
    df['ProductDiversityPerTrans'] = df['UniqueProducts'] / df['Frequency']

    # 5. Return/Complaint intensity (negative behavior normalized)
    df['ReturnIntensity'] = df['ReturnRatio'] * df['Frequency']  # Total returns weighted by activity


    df.drop(columns=['MonetaryTotal', 'Frequency', 'UniqueProducts', 'ReturnRatio', 'CustomerTenureDays', 'Recency', ], inplace=True)
    print(len(df.columns))

    return df


def main():
    # When running as a script, save the processed dataset
    df_processed = load_and_process_data()
    df_processed.to_csv("data/processed/retail_customers_processed.csv", index=False)
    print("Processed dataset saved to data/processed/retail_customers_processed.csv")
    df_processed.drop(columns=['CustomerID'], inplace=True)
    df_processed=fix_age_column(df_processed)
    df_processed=fix_support_tickets_and_satisfaction(df_processed)
    df_processed=fix_registration_date(df_processed)
    df_processed=fix_newsletter_subscribed(df_processed)
    df_processed=fix_AvgDaysBetweenPurchases(df_processed)
    df_processed=fix_account_status_and_churn(df_processed)
    
    df_processed=extract_ip_features(df_processed, ip_col="LastLoginIP")
    df_processed=encoding(df_processed)
    print(df_processed.columns)
    df_processed=feature_engineering(df_processed)
    print(df_processed.columns)


    df_processed.to_csv("data/processed/retail_customers_processed.csv", index=False)
    print("Processed dataset saved to data/processed/retail_customers_processed.csv")

    print(df_processed.isnull().sum())


if __name__ == "__main__":
    main()