#!/usr/bin/env python
from datetime import datetime
from time import time
from lxml import html,etree
from reviews_final import scrape, write_in_csv
import pandas as pd
import requests,re
import os,sys
import unicodecsv as csv
import argparse
import numpy as np
import json
def clean(text):
    if text:
        # Removing \n \r and \t
        return ' '.join(''.join(text).split()).strip()
    return None




def parse(locality,checkin_date,checkout_date,sort):
    checkIn = checkin_date.strftime("%Y/%m/%d")
    checkOut = checkout_date.strftime("%Y/%m/%d")
    print ("Scraper Inititated for Locality:%s"%locality)
    header = {

                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
            }
    # TA rendering the autocomplete list using this API
    print ("Finding search result page URL")
    geo_url = 'https://www.tripadvisor.com/TypeAheadJson?action=API&startTime='+str(int(time()))+'&uiOrigin=GEOSCOPE&source=GEOSCOPE&interleaved=true&types=geo,theme_park&neighborhood_geos=true&link_type=hotel&details=true&max=12&injectNeighborhoods=true&query='+locality
    api_response  = requests.get(geo_url,headers=header, timeout=120).json()
    #getting the TA url for th equery from the autocomplete response
    url_from_autocomplete = "http://www.tripadvisor.com"+api_response['results'][0]['url']
    print ('URL found %s'%url_from_autocomplete)
    geo = api_response['results'][0]['value']
    #Formating date for writing to file
    a=url_from_autocomplete
    b=a.split("-")
    s="-"
    c=s.join([b[0],b[1],"oa30",b[2],b[3]])
    d=s.join([b[0],b[1],"oa60",b[2],b[3]])
    e=s.join([b[0],b[1],"oa90",b[2],b[3]])
    f=s.join([b[0],b[1],"oa120",b[2],b[3]])
    urllist = [a,c,d,e,f]

    date = checkin_date.strftime("%Y_%m_%d")+"_"+checkout_date.strftime("%Y_%m_%d")
    #form data to get the hotels list from TA for the selected date
    form_data = {'changeSet': 'TRAVEL_INFO',
            'showSnippets': 'false',
            'staydates':date,
            'uguests': '2',
            'sortOrder':sort

    }



    json_arr = []
    for url_from_autocomplete in urllist:
        print(url_from_autocomplete)

        headers = {
                                'Accept': 'text/javascript, text/html, application/xml, text/xml, */*',
                                'Accept-Encoding': 'gzip,deflate',
                                'Accept-Language': 'en-US,en;q=0.5',
                                'Cache-Control': 'no-cache',
                                'Connection': 'keep-alive',
                                'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
                                'Host': 'www.tripadvisor.com',
                                'Pragma': 'no-cache',
                                'Referer': url_from_autocomplete,
                                'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:28.0) Gecko/20100101 Firefox/28.0',
                                'X-Requested-With': 'XMLHttpRequest'
                            }
        cookies=  {"SetCurrency":"USD"}
        print ("Downloading search results page")
        page_response  = requests.post(url = url_from_autocomplete,data=form_data,headers = headers, cookies = cookies, verify=False)
        print ("Parsing results ")
        parser = html.fromstring(page_response.text)
        hotel_lists = parser.xpath('//div[contains(@class,"listItem")]//div[contains(@class,"listing collapsed")]')
        hotel_data = []
        if not hotel_lists:
            hotel_lists = parser.xpath('//div[contains(@class,"listItem")]//div[@class="listing "]')

        for hotel in hotel_lists:
            XPATH_HOTEL_LINK = './/a[contains(@class,"property_title")]/@href'
            XPATH_REVIEWS  = './/a[@class="review_count"]//text()'
            XPATH_RANK = './/div[@class="popindex"]//text()'
            XPATH_RATING = './/span[contains(@class,"ui_bubble_rating bubble_45")]/@alt' #update this code to get rating
            XPATH_RATING_2 = './/a[contains(@class,"ui_bubble_rating bubble_45")]/@alt' #update this code to get rating
            XPATH_HOTEL_NAME = './/a[contains(@class,"property_title")]//text()'
            XPATH_HOTEL_FEATURES = './/div[contains(@casls,"common_hotel_icons_list")]//li//text()'
            XPATH_HOTEL_PRICE = './/div[contains(@data-sizegroup,"mini-meta-price")]/text()'
            XPATH_VIEW_DEALS = './/div[contains(@data-ajax-preserve,"viewDeals")]//text()'
            XPATH_BOOKING_PROVIDER = './/div[contains(@data-sizegroup,"mini-meta-provider")]//text()'  #<span class="dekGp Ci _R S4 H3 MD">#74 of 319 hotels in Lisbon</span><span class="dekGp Ci _R S4 H3 MD">#6 of 319 hotels in Lisbon</span>
            XPATH_RATING_ORDER = './/span[contains(@class,"dekGp Ci _R S4 H3 MD")]//text()'
            XPATH_OFFICIAL_DESCRIPTION = '//div[contains(text(),"Description")]/following-sibling::div//span[contains(@class,"introText")]/text()'


            raw_booking_provider = hotel.xpath(XPATH_BOOKING_PROVIDER)
            raw_no_of_deals =  hotel.xpath(XPATH_VIEW_DEALS)
            raw_hotel_link = hotel.xpath(XPATH_HOTEL_LINK)
            raw_no_of_reviews = hotel.xpath(XPATH_REVIEWS)
            raw_rank = hotel.xpath(XPATH_RANK)
            raw_rating = hotel.xpath(XPATH_RATING_2)
            raw_hotel_name = hotel.xpath(XPATH_HOTEL_NAME)
            raw_hotel_features = hotel.xpath(XPATH_HOTEL_FEATURES)
            raw_hotel_price_per_night  = hotel.xpath(XPATH_HOTEL_PRICE)
            raw_rank_order = hotel.xpath(XPATH_RATING_ORDER)
            raw_official_description = parser.xpath(XPATH_OFFICIAL_DESCRIPTION)

            url = 'http://www.tripadvisor.com'+raw_hotel_link[0] if raw_hotel_link else  None
            reviews = ''.join(raw_no_of_reviews).replace("reviews","").replace(",","") if raw_no_of_reviews else 0
            rank = ''.join(raw_rank) if raw_rank else None
            rating = ''.join(raw_rating).replace('of 5 bubbles','').strip() if raw_rating else None
            name = ''.join(raw_hotel_name).strip() if raw_hotel_name else None
            hotel_features = ','.join(raw_hotel_features)
            #price_per_night = ''.join(raw_hotel_price_per_night).encode('utf-8').replace('\n','') if raw_hotel_price_per_night else None
            price_per_night = ''.join(raw_hotel_price_per_night).replace('\n','') if raw_hotel_price_per_night else None
            rank_order = ''.join(raw_rank_order) if raw_rank_order else None
            no_of_deals = re.findall("all\s+?(\d+)\s+?",''.join(raw_no_of_deals))
            booking_provider = ''.join(raw_booking_provider).strip() if raw_booking_provider else None
            official_description = clean(raw_official_description)

            if no_of_deals:
                no_of_deals = no_of_deals[0]
            else:
                no_of_deals = 0

            data = {
                        'hotel_name':name,
                        'url':url,
                        'locality':locality,
                        'reviews':reviews,
                        'rank':rank,
                        'tripadvisor_rating':rating,
                        'checkOut':checkOut,
                        'checkIn':checkIn,
                        'hotel_features':hotel_features,
                        'price_per_night':price_per_night,
                        'no_of_deals':no_of_deals,
                        'booking_provider':booking_provider,
                        'raw_rank': rank_order,
                        'desc':official_description

            }


            if data:
                print("Writing scraped data")
                json_arr.append(data)
                with open('data_file.json', 'w') as outfile:
                    json.dump(json_arr, outfile)
    #         hotel_data.append(data)
    #         all_hotel.append(data)
    # #Referrer is necessary to get the correct response from TA if not provided they will redirect to home page
    # my_df=pd.DataFrame(all_hotel)
    # print(my_df['hotel_name'])





    return urllist
