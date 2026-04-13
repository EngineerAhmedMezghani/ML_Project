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

    print("----------------registration date------------------")
    print(df["RegistrationDate"].head())
    # Convert RegistrationDate to datetime with day first
    df["RegistrationDate"] = pd.to_datetime(
        df["RegistrationDate"], 
        format="%d/%m/%y",  # day/month/two-digit year
        errors="coerce"
    )
    df["RegistrationDate_day"] = df["RegistrationDate"].dt.day
    df["RegistrationDate_month"] = df["RegistrationDate"].dt.month
    df["RegistrationDate_year"] = df["RegistrationDate"].dt.year
    print(df[["RegistrationDate", "RegistrationDate_day", "RegistrationDate_month", "RegistrationDate_year"]])



    print("----------------NewsletterSubscribed------------------")
    df.drop(columns=["NewsletterSubscribed"], inplace=True)
    print(df.describe())
    print(len(df.columns))


    print("----------------LastLoginIP------------------")
    print("unique values:")
    print(len(df["LastLoginIP"].unique()))
    print(df["LastLoginIP"].head())

    print("----------------AccountStatus, Churn------------------")
    print(df.columns.unique())
    print(df["AccountStatus"].head(30))
    print("unique values:")
    print(len(df["AccountStatus"].unique()))
    print("possible values of AccountStatus:", sorted(df["AccountStatus"].unique()))
    print("frequencies of AccountStatus:")
    print(df["AccountStatus"].value_counts())


    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)

    # Panier moyen
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency']

    # Anciennete vs activite recente
    df['TenureRatio'] = df['Recency'] / df['CustomerTenureDays']


    df.drop(columns=["MonetaryTotal", "Recency", "Frequency", "CustomerTenureDays"], inplace=True)


    # print("----------------GeoIP Numeric Feature Engineering------------------")

    # import ipaddress

    # def ip_to_int(ip):
    #     try:
    #         return int(ipaddress.IPv4Address(ip))
    #     except:
    #         return None


    # def cidr_to_range(cidr):
    #     net = ipaddress.ip_network(cidr, strict=False)
    #     return int(net.network_address), int(net.broadcast_address)


    # # Load datasets
    # blocks = pd.read_csv("data/external/GeoLite2-Country-Blocks-IPv4.csv")
    # locations = pd.read_csv("data/external/GeoLite2-Country-Locations-en.csv")

    # # Keep only needed columns
    # blocks = blocks[["network", "geoname_id"]]
    # locations = locations[["geoname_id", "country_iso_code"]]

    # # Merge
    # geo_df = blocks.merge(locations, on="geoname_id", how="left")

    # # Convert CIDR → range
    # geo_df["ip_start"] = geo_df["network"].apply(lambda x: cidr_to_range(x)[0])
    # geo_df["ip_end"] = geo_df["network"].apply(lambda x: cidr_to_range(x)[1])


    # def get_country(ip):
    #     ip_int = ip_to_int(ip)
    #     if ip_int is None:
    #         return "UNK"

    #     match = geo_df[(geo_df["ip_start"] <= ip_int) & (geo_df["ip_end"] >= ip_int)]

    #     if match.empty:
    #         return "UNK"

    #     return match.iloc[0]["country_iso_code"]


    # # Map IP → country code
    # df["Country"] = df["LastLoginIP"].apply(get_country)

    # # Convert directly to numeric features
    # df["Country"] = df["Country"].astype("category")
    # df["Country_encoded"] = df["Country"].cat.codes

    # country_freq = df["Country"].value_counts(normalize=True)
    # df["Country_freq"] = df["Country"].map(country_freq)

    # # FINAL CLEANUP → remove redundancy
    # df.drop(columns=["LastLoginIP", "Country"], inplace=True)

    # print("Final numeric features added successfully")
    # print(df[["Country_encoded", "Country_freq"]].head())

    print("----------------Class imbalance check------------------")

    print("Churn distribution:")
    print(df["Churn"].value_counts(normalize=True))

    print("\nAccountStatus distribution:")
    print(df["AccountStatus"].value_counts(normalize=True))
    #we will drop the account status column
    df.drop(columns=["AccountStatus"], inplace=True)
    df.drop(columns=["CustomerID"], inplace=True)
    df.drop(columns=["RegistrationDate"], inplace=True)



    print("------------------------Encoding------------------------")
    print("-----------Gender Encoding-----------")

    mapping = {'M': 0, 'F': 1}
    # Application du mapping (Unknown deviendra NaN par défaut)
    df['Gender_encoded'] = df['Gender'].map(mapping)
    df.drop(columns=["Gender"], inplace=True)
    
    print("-----------Region Encoding-----------")
    mapping = {'UK': 0, 'Europe continentale': 1, 'Océanie': 2, 'Europe du Nord': 3, 'Autre': 4,
    'Europe centrale': 5, 'Europe du Sud': 5, "Europe de l'Est": 5, 'Asie': 4, 'Moyen-Orient': 4,
    'Amérique du Nord': 6, 'Amérique du Sud': 6, 'Afrique': 4}
    # Application du mapping (Unknown deviendra NaN par défaut)
    df['Region_encoded'] = df['Region'].map(mapping)
    df.drop(columns=["Region"], inplace=True)

    print("-----------Favorite Season Encoding-----------")
    print(df["FavoriteSeason"].value_counts())
    mapping = {'Automne': 0, 'Hiver': 1, 'Printemps': 2, 'Été': 3}
    df['FavoriteSeason_encoded'] = df['FavoriteSeason'].map(mapping)
    df.drop(columns=["FavoriteSeason"], inplace=True)

    print("-----------Preferred Time of Day Encoding-----------")
    print("PreferredTimeOfDay value counts:",df["PreferredTimeOfDay"].value_counts())
    mapping = {'Matin': 0, 'Midi': 1, 'Après-midi': 2, 'Soir': 3}
    df['PreferredTimeOfDay_encoded'] = df['PreferredTimeOfDay'].map(mapping)
    df.drop(columns=["PreferredTimeOfDay"], inplace=True)

    print("------------WeekendPreference-----------------------")
    print("WeekendPreference value counts:",df["WeekendPreference"].value_counts())
    mapping = {'Inconnu': 0, 'Semaine': 1, 'Weekend': 2}
    df['WeekendPreference_encoded'] = df['WeekendPreference'].map(mapping)
    df.drop(columns=["WeekendPreference"], inplace=True)

    print("-----------Customer Type Encoding-----------")
    print("CustomerType value counts:",df["CustomerType"].value_counts())
    mapping = {'Occasionnel': 0, 'Nouveau': 1, 'Perdu': 2, 'Régulier': 3, 'Hyperactif': 4}
    df['CustomerType_encoded'] = df['CustomerType'].map(mapping)
    df.drop(columns=["CustomerType"], inplace=True)

    print("----------------Country Encoding----------------")
    print("Country feature is dropped")
    df.drop(columns=["Country"], inplace=True)




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

    # 5. BasketSizeCategory: Size order (Petit=Small to Grand=Large)
    basket_mapping = {
        'Petit': 0,
        'Moyen': 1,
        'Grand': 2
    }
    df['BasketSizeCategory'] = df['BasketSizeCategory'].map(basket_mapping)

    # 6. ProductDiversity: Specialization level (Spécialisé=Focused to Explorateur=Explorer)
    diversity_mapping = {
        'Spécialisé': 0,   # Buys few product types
        'Modéré': 1,       # Moderate variety
        'Explorateur': 2   # Buys many different products
    }
    df['ProductDiversity'] = df['ProductDiversity'].map(diversity_mapping)

    # 7. RFMSegment: Customer value segments (Dormants=Low to Champions=High)
    rfm_mapping = {
        'Dormants': 0,     # Inactive
        'Potentiels': 1,   # Potential
        'Fidèles': 2,      # Loyal
        'Champions': 3     # Best customers
    }
    df['RFMSegment'] = df['RFMSegment'].map(rfm_mapping)

    print("✅ Manual encoding completed!")
    print("Checking for any unmapped (NaN) values...")
    for col in ['LoyaltyLevel', 'AgeCategory', 'SpendingCategory', 
                'ChurnRiskCategory', 'BasketSizeCategory', 'ProductDiversity', 'RFMSegment']:
        if df[col].isnull().sum() > 0:
            print(f"WARNING: {col} has {df[col].isnull().sum()} unmapped values!")
        else:
            print(f"✅ {col}: OK")
        
    # print("--------------testing dataset--------------")
    # print(df.shape)
    # print(df.dtypes)
    # print(df.isnull().sum().sum())

    return df


if __name__ == "__main__":
    # When running as a script, save the processed dataset
    df_processed = load_and_process_data()
    df_processed.to_csv("data/processed/retail_customers_processed.csv", index=False)
    print("Processed dataset saved to data/processed/retail_customers_processed.csv")
    # print(df_processed.head())
    # print("number of rows is :",len(df_processed.index))
    # print(df_processed.columns.unique())
    #print(df_processed["Churn"].value_counts())