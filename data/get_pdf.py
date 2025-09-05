import requests

def download_pdf(url,number):
    r = requests.get(url)
    with open(f"pdfs/{number}.pdf", "wb") as f:
        f.write(r.content)

for i in range(212, 389):
    if (i < 360):
        url = f"https://www.tsukuba.ac.jp/about/public-newspaper/{i}.pdf"
    else:
        url = f"https://www.tsukuba.ac.jp/about/public-newspaper/pdf/{i}.pdf"
    download_pdf(url, i)