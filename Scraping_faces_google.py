# Scraping_faces_google.py

from google_images_download import google_images_download as google
from selenium import webdriver
import ssl, os
ssl._create_default_https_context = ssl._create_unverified_context


def google_image(keyword, dir):
    response = google.googleimagesdownload()
    arguments = {'keywords': keyword,
                 'limit': 500,
                 'print_urls': False,
                 'no_directory' : True,
                 'output_directory': dir,
                 'chromedriver':'/home/cloudera/Documents/Develop/chromedriver'
                 }
    response.download(arguments)


def test(keywords):
    cwd = os.getcwd()
    for keyword in keywords:
        dir = cwd + f"/{keyword}"

        if not os.path.exists(dir):
            os.mkdir(dir)
        google_image(keyword, dir)


def main():
    keywords = ['bts정국','bts태형'] ## if needed change keywords
    test(keywords)

main()