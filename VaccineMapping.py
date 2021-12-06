import pandas as pd
import re
from collections import defaultdict
import csv
import sys

# a class to represent a particular OMOP vocabulary vaccine.
class VaccineConcept:
    def __init__(self, id, name):
        self.id = id   # OMOM vocabulary ID
        self.name = name
        self.id_name = id+' '+name
        self.mapped = set()    # standard concepts this concept is mapped to
        self.bow = set(re.split(' / | | , |\\|', name.lower()))  # set of words of name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, VaccineConcept):
            return self.id == other.id
        return False

    # this method adds a standard concept this concept is mapped to
    def add_mapped_con(self, mapped_con):
        self.mapped.add(mapped_con)


# returns a dictionary: key=id, value=VaccineConcept object
def load_vaccine_mappings_from_file(mapping_file_dataframe):
    vaccine_concept_map = {}    #key= concept id, value=VaccineConcept object
    for i in range(len(mapping_file_dataframe.index)):
        con_id = mapping_file_dataframe.iloc[i]['source_concept_id']
        con_name = mapping_file_dataframe.iloc[i]['source_concept_name']
        mapped_con = mapping_file_dataframe.iloc[i]['target_concept_id']

        if con_id not in vaccine_concept_map:
            con_obj = VaccineConcept(con_id, con_name)
            vaccine_concept_map[con_id] = con_obj
        con_obj = vaccine_concept_map[con_id]
        con_obj.add_mapped_con(mapped_con)

    return vaccine_concept_map


# from all vaccine concepts, create two seperate concepts with non-standard (source) and standard (target) concepts
def generate_source_target_concept_sets(mapping_file_dataframe):
    source_cons = set(mapping_file_dataframe[mapping_file_dataframe['source_standard_concept'] != 'S']['source_concept_id'])
    target_cons = set(mapping_file_dataframe[mapping_file_dataframe['source_standard_concept'] == 'S']['source_concept_id'])
    return source_cons, target_cons


# generates an ITP from two set-of-words
def generate_inferred_term_pair(con1_bow, con2_bow):
    return (frozenset(con1_bow.difference(con2_bow)), frozenset(con2_bow.difference(con1_bow))) # frozen set used since these needs to be used as dictionary keys


# generates ITP from existing mappings
def ITP_from_existing_mappings(source_con_set, vac_con_map):
    existing_mappings_ITPs = defaultdict(set)  # key=tuple:ITP, value= set:example mappings with ITP
    for source_con in source_con_set:
        source_con_obj = vac_con_map[source_con]

        for mapped_con in source_con_obj.mapped:
            mapped_con_obj = vac_con_map[mapped_con]
            itp = generate_inferred_term_pair(source_con_obj.bow, mapped_con_obj.bow)
            existing_mappings_ITPs[itp].add((source_con, mapped_con))

    return existing_mappings_ITPs


# Identifies potential mapping inconsistencies and writes it to csv file
def identify_mappings_inconsistency(source_con_set, target_con_set, vac_con_map, existing_mappings_ITPs, output_file):
    missing_mappings = defaultdict(set)     # key=tuple:ITP, value= set:missing mappings with ITP
    for i, source_con in enumerate(source_con_set):
        source_con_obj = vac_con_map[source_con]

        for target_con in target_con_set:
            if target_con in source_con_obj.mapped: # already mapped
                continue
            target_con_obj = vac_con_map[target_con]
            itp = generate_inferred_term_pair(source_con_obj.bow, target_con_obj.bow)
            if itp in existing_mappings_ITPs:
                missing_mappings[itp].add((source_con, target_con))

    with open(output_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Missing source ID', 'Missing source name', 'Missing target ID', 'Missing target name', 'Inferred Term Pair', 'Existing source ID', 'Existing source name', 'Existing target ID', 'Existing target name'])
        for itp, miss_maps in missing_mappings.items():
            itp1 = set(itp[0]) if len(itp[0]) > 0 else {}
            itp2 = set(itp[1]) if len(itp[1]) > 0 else {}
            for miss_map in miss_maps:
                missing_src_obj = vac_con_map[miss_map[0]]
                missing_trgt_obj = vac_con_map[miss_map[1]]
                existing = next(iter(existing_mappings_ITPs[itp]))
                existing_src_obj = vac_con_map[existing[0]]
                existing_trgt_obj = vac_con_map[existing[1]]
                if len(itp[0]) == 0 and len(itp[1]) == 0:
                    csvwriter.writerow(
                        [missing_src_obj.id, missing_src_obj.name, missing_trgt_obj.id, missing_trgt_obj.name, (itp1, itp2),
                         '', '', '', ''])
                else:
                    csvwriter.writerow(
                        [missing_src_obj.id, missing_src_obj.name, missing_trgt_obj.id, missing_trgt_obj.name, (itp1, itp2),
                         existing_src_obj.id, existing_src_obj.name, existing_trgt_obj.id, existing_trgt_obj.name])


def main():
    mapping_excel_file = sys.argv[1]
    output_csv_file = sys.argv[2]

    mapping_file_dataframe = pd.read_excel(mapping_excel_file, sheet_name=1)
    mapping_file_dataframe['source_standard_concept'] = mapping_file_dataframe['source_standard_concept'].fillna('')    # converting np.nan to empty string. Otherwise, causes issues
    mapping_file_dataframe = mapping_file_dataframe.astype('string')
    vac_con_map = load_vaccine_mappings_from_file(mapping_file_dataframe)
    print('Num vaccine concepts: ', len(vac_con_map))
    source_cons, target_cons = generate_source_target_concept_sets(mapping_file_dataframe)

    print('Num source cons: ', len(source_cons))
    print('Num target cons: ', len(target_cons))

    existing_mappings_ITPs = ITP_from_existing_mappings(source_cons, vac_con_map)
    print('Num ITPs - mapped: ', len(existing_mappings_ITPs))
    identify_mappings_inconsistency(source_cons, target_cons, vac_con_map, existing_mappings_ITPs, output_csv_file)


if __name__ == '__main__':
    main()
