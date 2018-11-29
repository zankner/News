import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

#pull down article links


class Data(object):

    def __init__(self, url, urlbase):
        self.url = url
        self.urlbase = urlbase


    def get_data(self, url, urlbase):
        urltopics = {}
        soup = BeautifulSoup(urllib.request.urlopen(url), features="html.parser")
        topics = soup.find_all('div', {'nav-menu-links'})
        for label in topics:
            for lb in label.contents:
                if not lb.get('href').startswith('http:') and not lb.get('href').startswith('/videos') and not lb.get('href').startswith('/style')\
                        and not lb.get('href').startswith('/travel'):
                    new_url = urlbase + lb.get('href')
                    urltopics[lb.string] = new_url
        return (urltopics)



    def get_articles(self, urltopics, urlbase):
        articles = {}
        for key in urltopics:
            soup = BeautifulSoup(urllib.request.urlopen(urltopics[key]), features="html.parser")
            links = soup.find_all('h3', {'cd__headline'})
            if key is "U.S":
                url_key = key.replace(".", "")
            url_key = key.lower()
            article_links = []
            for lb in links:
                for labl in lb.contents:
                    if labl.get('href').startswith("/2018") or labl.get('href').startswith(url_key):
                        article_links.append(urlbase + labl.get('href'))
                        articles[key] = article_links
        return (articles)

    def art(self):
        return(self.get_articles(self.get_data(self.url,self.urlbase),self.urlbase))





'''
dat = Data(url = "https://cnn.com"
    , urlbase = "https://cnn.com")
print ( dat.get_articles(urltopics=dat.get_data(dat.url, dat.urlbase), urlbase= dat.urlbase))
'''
