
from bs4 import BeautifulSoup
import requests
import os

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import logging
from time import sleep
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException
import shutil
import hashlib
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

#username = 'mary-anne.hartley@epfl.ch'
#password = 'lusst2021'
# username = 'noemie.boillat@chuv.ch'
# password = 'Optiresp'


username = 'mary-anne.hartley@epfl.ch'
password  = 'lusst2021'

#archivename = 'OPTI RESP'
archivename = 'CLUSSTER Bénin'
#download_dir = '/Users/trevorbrokowski/Desktop/TBLUScopy/CLUSSTER-Benin/CLUSSTERBeninVids'

download_dir = '/Users/trevorbrokowski/Desktop/VideoDownloads'
videofolder = 'Uncleaned_v2'

videofolderpath = os.path.join(download_dir, videofolder)
if not os.path.exists(videofolderpath):
    os.mkdir(videofolderpath)
current_downloadspath = os.path.join(videofolderpath, 'current_downloads3')
if not os.path.exists(current_downloadspath):
    os.mkdir(current_downloadspath)

options = Options()
prefs = {"download.default_directory": current_downloadspath, "download.prompt_for_download": False}
options.add_experimental_option('prefs', prefs)

driver = webdriver.Chrome(options=options)  

driver.get("https://cloud.butterflynetwork.com/")  


wait = WebDriverWait(driver, 20)

email_field = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@data-bni-id='emailField']")))
email_field.clear()
email_field.send_keys(username)

password_field = driver.find_element(By.XPATH, "//input[@data-bni-id='passwordField']")
password_field.clear()
password_field.send_keys(password)

login_button = driver.find_element(By.XPATH, "//button[@data-bni-id='loginButton']")
login_button.click()

link = WebDriverWait(driver, 20).until(
    EC.presence_of_element_located(
        (By.XPATH, "//span[contains(text(), '{}')]/ancestor::a".format(archivename))
    )
)
href = link.get_attribute('href')
link.click()

def extract_info_from_page():
    sleep(10)
    rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//tr[@data-bni-id='DataGridTableRow']")))
    for row in rows:
        try:
            cell = row.find_element(By.CLASS_NAME, 'DataGridTable-module--last-frozen--5Qjle')
            
            a_tag = cell.find_element(By.TAG_NAME, 'a')
            span_tag = cell.find_element(By.CLASS_NAME, "flex-grow.font-bold.truncate")
            
            href = a_tag.get_attribute('href')
            
            title = span_tag.text
            
            file_info.add((href, title))
        except Exception as e:
            print("An error occurred while extracting information from a row:", e)


def go_to_next_page():
    next_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@data-testid='next-btn']")))
    next_button.click()

def extract_video_urls_from_page():
    groups = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//div[contains(@class, 'relative rounded group w-[100px] h-[100px] AspectRatioBox-CssProp1_Component-module--cls2--5tOys AspectRatioBox-CssProp1_Component-module--cls1--6B+CF ')]")))
    videos = []
    for group in groups:
        inset_div = group.find_element(By.XPATH, ".//div[contains(@class, 'inset-0 absolute')]")
        a_tag = inset_div.find_element(By.TAG_NAME, 'a')
        href = a_tag.get_attribute('href')
        videos.append(href)
    return videos

def download_video(videos):
    for video_url in videos:
        driver.get(video_url)
        download_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@data-bni-id='DownloadButton']"))
            )
        download_button.click()
        submit_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
            )
        submit_button.click()

def download_video_attempt2(videos):
    for video_url in videos:
        try:
            # Navigate to the video page
            driver.get(video_url)
            logging.info(f"Navigating to video page: {video_url}")
            
            # Try to find and click the download button
            try:
                # Wait for the page to fully load
                sleep(3)
                
                # First, look for the download button
                download_button = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[@data-bni-id='DownloadButton']"))
                )
                logging.info("Found download button")
                download_button.click()
                logging.info("Clicked download button")
                
                # Then look for the submit/confirm button and click it
                submit_button = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
                )
                logging.info("Found submit button")
                submit_button.click()
                logging.info("Clicked submit button")
                
                # Wait for download to start and complete
                # This is a critical part - we need to wait long enough
                initial_files = set(os.listdir(current_downloadspath))
                
                # Wait for new files to appear (up to 30 seconds)
                max_wait = 30
                for i in range(max_wait):
                    sleep(1)
                    current_files = set(os.listdir(current_downloadspath))
                    new_files = current_files - initial_files
                    if new_files:
                        logging.info(f"Download completed after {i+1} seconds")
                        break
                    if i == max_wait - 1:
                        logging.warning("No new files detected after maximum wait time")
                
                # Check if download was successful
                if len(os.listdir(current_downloadspath)) > len(initial_files):
                    logging.info(f"Successfully downloaded video from {video_url}")
                else:
                    logging.warning(f"No new files found in download directory for {video_url}")
                    
            except Exception as e:
                logging.error(f"Error during UI download process: {e}")
                raise Exception(f"Download failed: {e}")
                
        except Exception as e:
            logging.error(f"Error downloading {video_url}: {e}")
        
# Collecting info from all pages
file_info = set()
while True:
    extract_info_from_page()
    try:
        go_to_next_page()
    except Exception as e:
        print("No more pages or an error occurred:", e)
        break
    
ids_to_gather =  ['25-628',
                '25-636',
                '25-638',
                '25-639',
                '25-640',
                '25-645',
                '25-648',
                '25-650',
                '25-651',
                '25-657',
                '25-658',
                '25-660',
                '25-664',
                '25-668',
                '25-670',
                '25-677',
                '25-679',
                '25-682',
                '25-687',
                '25-693',
                '25-696',
                '25-698',
                '25-709',
                '25-712',
                '25-713',
                '25-718',
                '25-725',
                '25-730',
                '25-735',
                '25-740',
                '25-743',
                '25-744',
                '25-746',
                '25-748',
                '25-751']


new_file_info = set()
    
for f in file_info:
    if f[1].split(',')[0] in ids_to_gather:
        new_file_info.add(f)
print(new_file_info)

import os
import logging
from time import sleep
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException
import shutil
logging.basicConfig(level=logging.INFO)


def download_and_rename_files(file_info, videofolderpath, current_downloadspath, max_retries=1):
    failed_downloads = []
    completed_downloads = []
    for href, title in tqdm(file_info, desc="Processing Files", unit="file", ncols=100):
        retries = 0
        if title:
            safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in ' -_']).rstrip()
            if safe_title in os.listdir(videofolderpath) or 'F' in safe_title:
                continue
        while retries < max_retries:
            try:
                driver.get(href)
                vids = extract_video_urls_from_page()
                download_video_attempt2(vids)  # Only use one download method
                break  
            except (URLError, WebDriverException) as e:
                logging.error(f"Error downloading {title} from {href}: {e}")
                retries += 1
                sleep(15)  
                continue
                
        # Remove the second download attempt (download_video_attempt2)
        
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in ' -_']).rstrip()
        new_path = os.path.join(videofolderpath, safe_title)
        if not os.path.exists(new_path):
            os.rename(current_downloadspath, new_path)
        else:
            logging.warning(f"Directory {new_path} already exists")
            
        if os.path.exists(current_downloadspath):
            sleep(20)
            shutil.rmtree(current_downloadspath)  
        os.mkdir(current_downloadspath)
        completed_downloads.append((href, title))
        logging.info(f"Prepared {current_downloadspath} for the next download.")
    
    logging.info("finished download")
    print(failed_downloads, completed_downloads)
    return failed_downloads, completed_downloads


print(file_info)
failed, completed = download_and_rename_files(new_file_info, videofolderpath, current_downloadspath)
driver.quit()