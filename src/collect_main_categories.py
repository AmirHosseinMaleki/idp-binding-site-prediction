import requests

terms = [
    "Protein binding",
    "DNA binding", 
    "RNA binding",
    "Ion binding"
]

for term in terms:    
    url = f"https://disprot.org/api/search?go_name={term.replace(' ', '%20')}&format=tsv&release=2022_03&page=0&sort_field=disprot_id&sort_value=asc"
    
    try:
        response = requests.get(url)
        
        safe_name = term.replace(' ', '_').replace('/', '_').lower()
        filename = f"data/{safe_name}_raw.tsv"
        
        with open(filename, 'w') as f:
            f.write(response.text)
        
        lines = response.text.split('\n')
        print(f"Saved {len(lines)-1} entries to: {filename}")
        
    except Exception as e:
        print(f"Error with {term}: {e}")