import os
import time
import urllib.request
from argparse import ArgumentParser

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def run(query, count):

    createFolder('./'+query+'/')

    options = webdriver.FirefoxOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    browser = webdriver.Firefox(options=options)
    #browser.set_window_size(1024, 768)

    url = 'https://yandex.ru/images/search?text='+query

    browser.get(url)
    time.sleep(1)

    element = browser.find_element(By.TAG_NAME, "body")
    # Scroll down
    for i in range(count//10):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id("smb").click()
        for i in range(count//10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection
    except:
        for i in range((count//100)+1):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)  # bot id protection

    print("Reached end of the page")
    time.sleep(0.5)

    source = browser.page_source
    soup = BeautifulSoup(source, 'lxml')  # choose lxml parser
    # find the tag : <img ... >
    image_tags = soup.findAll('img')
    # print out image urls
    counter = 0
    for image_tag in image_tags:

        if counter == count:
            break

        link = image_tag.get('src')

        if link is None or link == '':
            continue

        link = 'http:' + link
        print(link, counter+1)  # page source

        for i in range(3):
            try:
                urllib.request.urlretrieve(link, "./"+query+"/" + query + str(counter+1) + ".jpeg")
                counter += 1
                break
            except:
                print("\tSleeping..")
                time.sleep(2)
        time.sleep(0.1)


if __name__ == "__main__":
    imgCount = 300
    run(query="bicycle", count=imgCount)
