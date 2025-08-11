import requests

accession = "P03045"
url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"

print(f"URL: {url}")

try:
    response = requests.get(url)
    print(f"Success!{len(response.text)} characters")
    
    print(f"\n response:")
    print("-" * 50)
    print(response.text)
    print("-" * 50)
    
    filename = f"data/{accession}_sequence.fasta"
    with open(filename, 'w') as f:
        f.write(response.text)
    
except Exception as e:
    print(f"Error: {e}")