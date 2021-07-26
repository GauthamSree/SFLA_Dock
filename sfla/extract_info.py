import os

if __name__ == "__main__":
    result_path = "Result/"
    list_of_protein = os.listdir(result_path)
    list_of_files = [f"{result_path}{protein}/report_{protein}_0.txt" for protein in list_of_protein]
    protein_count = {}
    total_count = [0, 0, 0]

    for file_path, protein in zip(list_of_files, list_of_protein):
        with open(file_path, 'r') as report:
            
            flag_list = [False, False, False]
            for line in report:
                l=line.split()
                if len(l) == 5:
                    if l[4] == 'Acceptable' and not flag_list[0]:
                        flag_list[0] = True
        
                    if l[4]=='Medium' and not flag_list[1]:
                        flag_list[1] = True
        
                    if l[4]=='High' and not flag_list[2]:
                        flag_list[2] = True

            if flag_list[0] and not flag_list[1] and not flag_list[2]:
                total_count[0] += 1
            
            if flag_list[1] and not flag_list[2]:
                total_count[1] += 1

            if flag_list[2]:
                total_count[2] += 1
            
            protein_count[protein] = flag_list

    with open("report_final.txt", 'w') as file:
        for protein, flag_list in protein_count.items():
            line = f"{protein}:: [Acceptable = {flag_list[0]}, Medium = {flag_list[1]}, High = {flag_list[2]}]\n"
            file.write(line)
        
        final_count = f"\n\nTotal Count::[Acceptable = {total_count[0]}, Medium = {total_count[1]}, High = {total_count[2]}]\n"     
        total_sum = f"Total sum = {sum(total_count)} in {len(list_of_protein)}\n"
        file.write(final_count)
        file.write(total_sum)
        print(total_sum)
