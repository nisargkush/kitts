# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:59:51 2021

@author: nkushwaha
"""

#imports here
import time 
import argparse
#import requests, urllib.request
#import re
import json
import pandas as pd
import os
import wget
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from datetime import datetime
from bs4 import BeautifulSoup
from datetime import datetime
from config import PROJECT_ROOT_DIR, DATA_DIR, RAW_DATA_DIR, WEBDRIVER_LOC, BASE_URL, CSV_DIR

"""
class UgcData():
    def __init__(self, username ,userpass ,basepath ,web_url = 'http://www.instagram.com'
                 ,webDriver_path = 'C:/Docs/chromedriver_win32/chromedriver.exe'
                 ,hashtag = None):
        self.username = username
        self.userpass = userpass
        if hashtag == None:
            self.hashtag = []
        else:
            self.hashtag = hashtag
        self.basepath = basepath
        self.webDriver_path = webDriver_path
        self.web_url = 'http://www.instagram.com'
"""  

def collect_posts(user, userpass, hashtag, base_path):
    #specify the path to chromedriver.exe (download and save on your computer)
    driver = webdriver.Chrome(WEBDRIVER_LOC)#'C:/Docs/chromedriver_win32/chromedriver.exe'
    file_name =  ''    
    #open the webpage        
    print("Opening Browser")
    try:        
        driver.get(BASE_URL)
    
        #target username
        username = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='username']")))
        password = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='password']")))
    
        #enter username and password
        username.clear()
        username.send_keys(user)
        password.clear()
        password.send_keys(userpass)
    
        #target the login button and click it
        button = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))).click()
    
        #We are logged in!
        
        #nadle NOT NOW
        not_now = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Not Now")]'))).click()
        not_now2 = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Not Now")]'))).click()
        
        #target the search input field
        searchbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='Search']")))
        searchbox.clear()
    
        #search for the hashtag cat
        keyword = "#" + hashtag
        searchbox.send_keys(keyword)
     
        # Wait for 5 seconds
        time.sleep(5)
        searchbox.send_keys(Keys.ENTER)
        time.sleep(5)
        searchbox.send_keys(Keys.ENTER)
        time.sleep(5)
     
        #scroll down to scrape more images
        driver.execute_script("window.scrollTo(0, 5000);")
        time.sleep(10)
        #posts
        posts = []
        links = driver.find_elements_by_tag_name('a')
        for link in links:
            post = link.get_attribute('href')
            if '/p/' in post:
                posts.append(post)
    
        print('Number of scraped posts: ', len(posts))
        #return [driver, hashtag, posts]
        
    
        #collecting post details
        listinformation = []
        column_list = ['shortcode', 'post_date', 'post_url', 'display_url',
                   'like_and_view_counts_disabled','likes',
                   'is_paid_partnership','location','city','exact_city_match',
                   'is_video','is_verified','is_private']
        counter = 0
        for iteration in posts:
            html = driver.get(iteration + "?__a=1")
            soup = BeautifulSoup(driver.page_source, "html.parser").get_text()
            jsondata = json.loads(soup)
          
            shortcode = jsondata["graphql"]["shortcode_media"]["shortcode"]        
            post_date = datetime.fromtimestamp(jsondata["graphql"]["shortcode_media"]["taken_at_timestamp"]).strftime('%d-%m-%y')
            display_url = jsondata["graphql"]["shortcode_media"]["display_url"]
            like_and_view_counts_disabled = jsondata["graphql"]["shortcode_media"]["like_and_view_counts_disabled"]
            edge_media_preview_like = jsondata["graphql"]["shortcode_media"]["edge_media_preview_like"]["count"]
            is_paid_partnership = jsondata["graphql"]["shortcode_media"]["is_paid_partnership"]
            try:
                location = jsondata["graphql"]["shortcode_media"]["location"]["name"]
            except:
                location = ''
            try:
                address_j = json.loads(jsondata["graphql"]["shortcode_media"]["location"]["address_json"])
                city = address_j['city_name']
                exact_city_match = address_j['exact_city_match']
            except:
                city = ''
                exact_city_match = ''
            #has_public_page = jsondata["graphql"]["shortcode_media"]["location"]["has_public_page"]
            is_video = jsondata["graphql"]["shortcode_media"]["is_video"]
            is_verified = jsondata["graphql"]["shortcode_media"]["owner"]["is_verified"]
            is_private = jsondata["graphql"]["shortcode_media"]["owner"]["is_private"]
         
            listinformation.append([shortcode,post_date,iteration,display_url,
                                    like_and_view_counts_disabled,edge_media_preview_like,
                                    is_paid_partnership,location,city,exact_city_match,
                                    is_video,is_verified,is_private])
            counter += 1
        
        # Create Datafram and convert it to csv file based on given path    
        print('no of posts succesfully collected:',counter)   
        post_df = pd.DataFrame(listinformation,  columns = column_list)
        try:
            #print(base_path)
            file_path = os.path.normpath(os.path.join(DATA_DIR,hashtag))            
            if os.path.isdir(file_path):
                print(f'Path: {file_path} ,to store file already exists')
            else:
                os.mkdir(file_path)
            file_name = os.path.join(file_path,hashtag) + ".csv"
            if os.path.isfile(file_name):
                post_df.to_csv(file_name, index = None, mode='a', header=False)
            else:            
                post_df.to_csv(file_name, index = None)
        except:
            print("exception creating csv file")
    except Exception as e:
        print(e)
    finally:      
        print("Closing Browser")
        driver.close()
                
        return file_name
        
def collect_post_image(file):
    
    file_path, file_name= os.path.split(file)
    print("file_path",file_path)
    print("file_name",file_name)  
    image_path = os.path.normpath(os.path.join(file_path,'Images'))   
    try:  
        if os.path.isdir(image_path):
            pass
        else:
            os.mkdir(image_path)
        post_df = pd.read_csv(file)
        img_path_list=[]
        counter = 0
        for index, row in post_df.iterrows():
            save_as = os.path.join(image_path, row['shortcode'] + '.jpg')
            if os.path.isfile(save_as):                
                img_path_list.append(save_as)
            else:
                try:
                    wget.download(row['display_url'], save_as)
                    img_path_list.append(save_as)
                    counter += 1
                except:
                    img_path_list.append('')
        post_df["img_path"] = img_path_list 
        img_file = os.path.join(RAW_DATA_DIR,file_name)   
        post_df.to_csv(img_file, index = None)
        print('no of images succesfully downloaded:',counter)    
    except:
        print("exception creating csv file. Hence not downloaded images")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data collection from Instagram Post data based on hashtag')
    parser.add_argument('-u', '--username', type=str, default='narenkishu', required=False,
                        help='Login username')
    parser.add_argument('-p', '--password', type=str, default='Inst@1822', required=False,
                        help='Login Password')
    parser.add_argument('-t', '--hashtag', type=str, default=None, required=True,
                        help='Hashtag value without #')
    parser.add_argument('-b', '--basepath', type=str, default=DATA_DIR, required=False,
                        help="basepath to collect data")
    
    args = parser.parse_args()
    
    filename = ''
    filepath = os.path.normpath(os.path.join(args.basepath,args.hashtag))
    
        
    #Checking existence of files for given hastag and calling collect_post funtion
    if os.path.isdir(filepath):
        print(f'Path: {filepath} ,to store file already exists')
    else:
        filename = collect_posts(args.username, args.password, args.hashtag, args.basepath)
    
    #Checking success of collect_post funciton for given hastag and calling collect_post_image funtion
    if filename == '':
        print(f'Images already exist for tag #{args.hashtag}. Nothing to download')
    else:
        collect_post_image(filename)


