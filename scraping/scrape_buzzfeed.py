import grequests
import json
import multiprocessing
from tqdm import tqdm
import threading
from html import unescape

write_buffer = multiprocessing.Queue()

def writer_thread():
	with open('buzzfeed_dataset.txt', 'a+') as out:
		count = 0
		s = write_buffer.get(block=True)
		while s:
			out.write(s + '\n')
			count += 1
			if not count % 1000:
				out.flush()
				print ('Scraped and added {} titles...'.format(count))
			s = write_buffer.get()

t = threading.Thread(target=writer_thread)
t.daemon = True
t.start()

def scrape(http_response):
	feed = json.loads(http_response.text)
	for i in feed['big_stories']:
		write_buffer.put(unescape(i['title']), block=True)
	for i in feed['buzzes']:
		write_buffer.put(unescape(i['title']), block=True)
	
n_workers = 10
n_pages = 1000
pool = multiprocessing.Pool(n_workers)
buzzfeed_scrape = grequests.imap([grequests.get('http://www.buzzfeed.com/api/v2/feeds/index?p={}'.format(p)) for p in range(4001,n_pages+4001)])
pool.map(scrape, buzzfeed_scrape)

print('Scraping done')
write_buffer.put(None)

