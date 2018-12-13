import bag_process
import links
import get_text

def main():

    list_retr = links.Data(url = "https://cnn.com", urlbase = "https://cnn.com")
    art_dict=list_retr.art()
    text_retr = get_text.get_text(art_dict=art_dict)
    text_retr.perform()

if __name__ == '__main__':

    main()
