import yt_dlp
import os
from bs4 import BeautifulSoup
from requests_html import HTMLSession

news_base_url = 'https://vod.tvp.pl/informacje-i-publicystyka,205/wiadomosci-odcinki,273726'
base_url = 'https://vod.tvp.pl'

def load_page(url):
    """Load competition page with results and event data, but without Selenium becouse it's slow."""

    session = HTMLSession()
    response = session.get(url)
    response.html.render(timeout=10, sleep=0.5)

    return BeautifulSoup(response.html.html, 'html.parser')

def get_latest_tvp_news_url():
    page = load_page(news_base_url)

    latest_news_endpoint = page.find_all('a', {'class': 'tile__link'})[0].get('href')
    return base_url + latest_news_endpoint


def download_video():
    output_path = os.path.join(os.path.abspath(os.getcwd()), 'data', '%(title)s.%(ext)s')
    ydl_opts = {
        'format': 'bestvideo[height<=480]/best',
        'outtmpl': output_path
    }

    latest_news_url = get_latest_tvp_news_url()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(latest_news_url)

