import requests as rq
import pandas as pd
import time
import csv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
import os

# read excel file
df = pd.read_excel("Etf List.xlsx")
df = pd.DataFrame(df)

# 將etf的名字存入name_list
name_list = []
etf_num = 0
for index, row in df.iterrows():
    etf_num += 1
    name_list.append(row['Symbol'])

# 將網站的url存入url_list中
url_list = []
for i in range(0, etf_num):
    url_list.append("https://finance.yahoo.com/quote/" + name_list[i] + "/history?period1=1394812800&period2=1552579200&interval=1d&filter=history&frequency=1d")

# 使用chrome-headless模式
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--dns-prefetch-disable')


for i in range(len(url_list)):
    browser = webdriver.Chrome(chrome_options = chrome_options)
    # 設定10秒的implicitly wait
    browser.implicitly_wait(10)
    browser.get(url_list[i])

    # 利用xpath定位"Download"按鈕
    element = browser.find_element_by_xpath("//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[2]/span[2]/a")
    element.click()
    
    browser.close()

