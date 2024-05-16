import requests
import pandas as pd
import os

def fetch_animal_images(limit=38):
    URL = "https://commons.wikimedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": "Animal",
        "srnamespace": "6",
        "srlimit": limit
    }

    response = requests.get(URL, params=PARAMS)
    data = response.json()
    search_results = data['query']['search']

    images = []
    for item in search_results:
        title = item['title']
        if not title.lower().endswith(('.svg', '.ogv', '.ogg')):
            file_info_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={title}&prop=imageinfo&iiprop=url|extmetadata&format=json"
            file_response = requests.get(file_info_url)
            file_data = file_response.json()
            pages = file_data['query']['pages']
            for page_id in pages:
                page = pages[page_id]
                if 'imageinfo' in page:
                    image_info = page['imageinfo'][0]
                    image_url = image_info['url']
                    image_description = image_info['extmetadata']['ImageDescription']['value'] if 'ImageDescription' in image_info['extmetadata'] else "No description"
                    if download_image(image_url, page_id):
                        images.append({
                            'id': page_id,
                            'visual': image_url,
                            'caption': image_description
                        })
                    else:
                        print(f"Failed to download or verify image: {image_url}")
    return images


from PIL import Image

def download_image(image_url, image_id):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(image_url, headers=headers)
        if response.status_code == 200:
            file_path = os.path.join('../data/raw/images', f'{image_id}.jpg')
            with open(file_path, 'wb') as f:
                f.write(response.content)
            # Tenter d'ouvrir l'image avec Pillow pour vérifier sa validité
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Vérifie que l'image n'est pas corrompue
                return True
            except (IOError, SyntaxError):
                os.remove(file_path)  # Supprimer le fichier corrompu
                return False
        else:
            print(f"Error downloading image {image_id}: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception occurred while downloading image {image_id}: {e}")
        return False

def save_to_csv(images, file_path):
    df = pd.DataFrame(images)
    df.to_csv(file_path, index=False, encoding='utf-8') 

def main():
    images_directory = '../data/raw/images'
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    images = fetch_animal_images()
    data_directory = '../data/raw'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    file_path = os.path.join(data_directory, 'animal_images_wikimedia.csv')
    save_to_csv(images, file_path)
    print(f"Le fichier CSV a été enregistré dans {file_path}")
    print(f"Les images ont été téléchargées dans le dossier {images_directory}")

if __name__ == "__main__":
    main()
