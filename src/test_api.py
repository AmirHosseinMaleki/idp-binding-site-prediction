import requests


url = "https://disprot.org/api/search?go_name=Protein%20binding&format=tsv&release=2022_03&page=0&sort_field=disprot_id&sort_value=asc"


response = requests.get(url)


try:
    response = requests.get(url)
    
    filename = "data/protein_binding_raw.tsv"
    
    with open(filename, 'w') as f:
        f.write(response.text)
    
    
except Exception as e:
    print(f"Something went wrong: {e}")