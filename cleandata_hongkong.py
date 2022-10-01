import pandas as pd
import numpy as np
import csv

def readHotels():
    contents = []
    file = open(r"C:\Users\Downloads\HotelListInHong Kong.csv", "r", encoding="mbcs")
    csv_reader = csv.reader(file)
    for row in csv_reader:
        contents.append(row)
    format_rows = []
    for row in contents:
        # print(row)
        # break
        hotel_name = row[1]
        url = row[2]
        locality = row[3]
        reviews = row[4]
        checkin = row[6]
        checkout = row[7]
        price = row[8]
        provider = row[9]
        no_deals = row[10]
        format_rows.append([hotel_name, url, locality, reviews, checkin, checkout, price, provider, no_deals])
    data = pd.DataFrame(format_rows)
    data.columns = ['hotel_name', 'url', 'locality', 'reviews', 'checkIn', 'checkOut', 'price_per_night', 'booking_provider', 'no_of_deals']
    return data

def readReviews():
    contents = []
    dict = {}
    file = open(r"C:\Users\Downloads\hotelReviewsInHong Kong.csv", "r", encoding="mbcs")
    csv_reader = csv.reader(file)
    for row in csv_reader:
        contents.append(row)
    format_rows = []
    header = True
    for row in contents:
        if header:
            header = False
            continue
        hotel_name = ((row[3].split("    "))[1]).replace("\n","")
        hotelUrl = row[4]
        text = row[1]
        if (hotel_name in dict):
            dict[hotel_name].append([hotelUrl, text])
        else:
            dict[hotel_name] = [[hotelUrl, text]]
    
    for k,v in dict.items():
        text = ""
        hotelName = k
        hotelUrl = v[0][0]
        for innerList in v:
            text = innerList[1]
            format_rows.append([hotelName, hotelUrl, text])
    data = pd.DataFrame(format_rows)
    data.columns = ['hotel_name', 'hotel_url', 'all_review']
    return data

def generateCleanDataCSV():
    hotel_df = readHotels()
    print(hotel_df.head())
    review_df = readReviews()
    print(review_df.head())
    print(len(review_df))
    final_df = pd.merge(hotel_df, review_df, on="hotel_name")
    final_df.to_csv(r"C:\Users\Downloads\clean_data.csv", index = False)
    print(len(final_df))

# pd1 = readReviews()
# print(pd1.head())

# pd2 = readHotels()
# print(pd2.head())

generateCleanDataCSV()
