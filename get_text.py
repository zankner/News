from bs4 import BeautifulSoup
import requests

class get_text(object):

    def __init__(self, art_dict):
        self.art_dict = art_dict
        self.text = []
        self.output = []

    def populate(self, key, value):
            for urlbase in value:
                r = requests.get(urlbase)
                print(r.status_code)
                r = r.content
                soup = BeautifulSoup(r, features = "html.parser")
                soup = soup.find_all(attrs={'class':'zn-body__paragraph'})
                text =  ''
                for s in soup:
                    text = text + s.text
                    text = text.replace('\n', '').replace('\r', '')
                    text = text.replace(',', '')
                self.text.append(str(text))
                self.output.append(str(key))

    def to_csv(self):
        for t in self.text:
            print(t)
        with open('data.csv','w') as file:
            for t,o in zip(self.text, self.output):
                file.write(t+',')
                file.write(o)
                file.write('\n')

    def perform(self):
        for key, value in self.art_dict.items():
            self.populate(key,value)
        self.to_csv()
