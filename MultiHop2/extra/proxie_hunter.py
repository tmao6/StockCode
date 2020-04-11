from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

import bs4
import time

import sys


url = 'http://pubproxy.com/api/proxy?google=true'

proxieList = []

fh  = open('proxie_list.txt', 'a+')

while True:
    try:
        driver = webdriver.Chrome(ChromeDriverManager().install())  # install the webdriver
        driver.get(url) #link to use

        time.sleep(5) #NEEDED for the javascript to run in the browser and make the tables

        elem = driver.find_element_by_xpath("//*") #extracts the whole HTML file
        source_code = elem.get_attribute("outerHTML")
        cleanPageData = bs4.BeautifulSoup(source_code, 'html.parser') #parses the HTML

        longText = str(cleanPageData.text)

        print(longText)

        ip = longText[20:35]
        secondText= longText[35:]
        for char in secondText:
            if char in ['0','9','8','7','6','5','4','3','2','1']:
                ip = ip+(str(char))
            else:
                print("a")
                break

        if ip not in proxieList:
            proxieList.append(ip)
            fh.write(ip)
            fh.write('\n')
        else:
            print("b")
            fh.close()
            driver.close()
            driver.quit()
            break




    except Exception as e:
        print(e)
        print('c')
        fh.close()
        driver.close()
        driver.quit()
        break


