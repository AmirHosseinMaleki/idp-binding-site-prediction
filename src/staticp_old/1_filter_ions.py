#!/usr/bin/env python3

import csv
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

ions = []

with open('ligand.tsv', 'r') as f:
    csv_reader = csv.reader(f, delimiter='\t')
    next(csv_reader)
    
    for row in csv_reader:
        if len(row) < 5:
            continue
            
        ligand_name = row[0]
        smiles = row[4].split(';')[0]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                charge = Chem.GetFormalCharge(mol)
                atoms = mol.GetNumAtoms()
                
                if charge != 0 and atoms < 2:
                    ion_code = ligand_name.replace('ION ', '')
                    ions.append(ion_code)
                    print(f"{ion_code}: {smiles}")
        except:
            continue

with open('ions.txt', 'w') as f:
    for ion in ions:
        f.write(ion + '\n')