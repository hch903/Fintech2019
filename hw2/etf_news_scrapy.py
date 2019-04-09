from bs4 import BeautifulSoup
import requests
import json

page_list = ['/sections/features-and-news']
href_list = []
title_list = []


for i in range(1,10):
    page_list.append("/sections/features-and-news?page=" + str(i))

main_url = "https://www.etf.com"

for i in range(0,10):
    # 搜尋10頁的新聞
    url = main_url + page_list[i]

    # 對網站發出requests
    r = requests.get(url)
    
    # 轉成beautiful soup
    soup = BeautifulSoup(r.text, "html.parser")
    
    # 找到該頁的新聞標題以及超連結
    for h2_tag in (soup.find_all('h2')):
        for a_tag in h2_tag:
            href_list.append(a_tag.get('href'))
            title_list.append(a_tag.text)

news_num = len(href_list)
content_list = [''] * news_num

for i in range(len(href_list)):
    url = main_url + href_list[i]
    
    # 對網站發出requests
    r = requests.get(url)
    
    # 轉成beautiful soup
    soup = BeautifulSoup(r.text, "html.parser")
    

    # 處理文章需要換頁的情況
    if(soup.find('a', class_ = "fullArticleLnk")):
        full_article = soup.find('a', class_ = "fullArticleLnk")
        full_article_href = full_article.get('href')
        url = main_url + full_article_href
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")

    # 找到article body的div tag
    div = soup.find("div", id = "article-body-content")
    p_tag = div.find_all('p')
    for content in p_tag:
        content_list[i] += content.text

title_and_content = dict()
output = dict()
for i in range(news_num):
    title_and_content[i] = dict()

for i in range(news_num):
    title_and_content[i] = {'topic': title_list[i], 'parag': content_list[i]}
    output[i] = title_and_content[i]

# 輸出json檔
with open('output.json', 'w') as fp:
    json.dump(output, fp, indent = 1)