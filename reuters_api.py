import os
import requests

from dotenv import load_dotenv

load_dotenv()
env = os.environ

reutersAPIUrl = env["REUTERS_API_URL"]
apiKey = env["REUTERS_API_KEY"]
apiHost = env["REUTERS_API_HOST"]
headers = {
    "X-RapidAPI-Key": apiKey,
    "X-RapidAPI-Host": apiHost
}


def getArticlesByKeywordName(keyword, pageNumber: int, limitPerPage: int):
    url = f"{reutersAPIUrl}/get-articles-by-keyword-name/{keyword}/{pageNumber}/{limitPerPage}"

    response = requests.get(url, headers=headers)
    return response.json()
