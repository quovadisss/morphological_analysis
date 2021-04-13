import json
import requests
import datetime
import random
import time
import re
import pandas as pd
import numpy as np
import argparse
from collections import Counter

from urllib.request import urlopen
from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


# Get news urls from thefintechtimes
def fintech_urls():
    driver = Chrome('/Users/mingyupark/spyder/chromedriver')
    first = 'https://thefintechtimes.com/category/fintech/'
    driver.get(first)
    urls = []
    for i in range(300):
        articles = driver.find_elements_by_css_selector('h2.entry-title > a')
        url = []
        for i in articles:
            if 'Latest Fintech Jobs' not in i.text:
                url.append(i.get_attribute('href'))
        urls.extend(url)
        try:
            next_page = driver.find_elements_by_css_selector('a.next.page-numbers')[0]
            driver.get(next_page.get_attribute('href'))
        except IndexError:
            break
    driver.quit()
    return urls


# Get each news contents for fintech
def fintech_text():
    urls = fintech_urls()
    titles = []
    contents = []
    for i in urls:
        driver.get(i)
        # Title of the news
        selector = 'h1.entry-title.penci-entry-title.penci-title-'
        title_css = driver.find_elements_by_css_selector(selector)
        titles.append(title_css[0].text)
        # Contents of the news
        selector = 'div.penci-entry-content.entry-content > p'
        content_css = driver.find_elements_by_css_selector(selector)
        content = ''
        for j in content_css:
            content += ' '
            content += j.text
        contents.append(content)

    df = pd.DataFrame(np.array([titles, contents]).T, columns=['title', 'news'])
    return df


# Get titles and urls from healthcareitnews
def health_url_title(pages):
    list_urls = []
    list_titles = []
    
    for i in pages:
        driver.get('https://www.healthcareitnews.com/search/content/healthcare?page={0}'.format(i)) 
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        nameList = soup.findAll("div",{"property":"dc:title"})
        
        for li in nameList:
            list_urls.append(li.find('a').get('href'))
            list_titles.append(li.find('a').text)
    return list_titles, list_urls


# Get each news contents for healthcare
def health_text(pages):
    titles, urls = health_url_title(pages)
    healthcare_news = []
    num = 0
    for title, url in zip(titles, urls):
        driver.get('https://www.healthcareitnews.com{0}'.format(url))
        time.sleep(1)
        document = driver.find_elements_by_css_selector('div.field-item.even > p')
        doc_text = ''
        for i in document:
            text = i.text
            if "Email the writer" not in text:
                doc_text += ' '
                doc_text += text
        if len(doc_text) > 5:
            healthcare_news.append((title, url,doc_text))
        num += 100
        print('{} pages done'.format(num))
    return healthcare_news


# Set parser to choose data set
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='healthcare',
                    help='You can use two datasets')
args = parser.parse_args()

# Execute chrome driver
driver = Chrome('/Users/mingyupark/spyder/chromedriver')

if args.data == 'healthcare':
    # Crawl healthcare IT news
    """
    Total page length must be more than 100.
    The result will be crawled by 100 pages to prevent errors.
    """
    print('Start crawling healthcare it news')
    total_page_length = 500
    cri = [i*100 for i in range(int(total_page_length/100)+1)]
    health_df = pd.DataFrame()
    for i in range(len(cri)-1):
        result = health_text(range(cri[i], cri[i+1]))
        df = pd.DataFrame(result, columns=['title', 'url', 'text'])
        health_df = pd.concat([health_df, df], axis=0)

    health_df.to_csv('data/health_news.csv', index=False)
    print('Crawling done')

else:
    # Crawl Mobile Payment news
    print('Start crawling Mobile payment it news')
    fintech_df = fintech_text()
    fintech_df.to_csv('data/fintech_news.csv', index=False)
    print('Crawling done')



