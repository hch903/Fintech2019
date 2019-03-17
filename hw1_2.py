from selenium import webdriver
import pandas as pd
m_driver = webdriver.Chrome()

m_driver.implicitly_wait(20)

m_driver.get("https://fred.stlouisfed.org/series/BAMLH0A3HYCEY")


dw_btn = m_driver.find_element_by_xpath('//*[@id="download-button"]')
dw_btn.click()
excel_btn = m_driver.find_element_by_xpath('//*[@id="fg-download-menu"]/li[1]/a')

url = excel_btn.get_attribute('href')
df = pd.read_excel(url)

df = df.drop(df.index[[0,1,2,3,4,5,6,7,8,9]])
df = df.reset_index(drop = True)
df['FRED Graph Observations'] = pd.to_datetime(df['FRED Graph Observations'], format="%Y%m%d")
df = df.rename(columns = {'FRED Graph Observations':'Date'})
df = df.rename(columns = {'Unnamed: 1':'Value'})
look = df.head(20)
print(look)