import pandas as pd
import openpyxl

# Excels file set up
input_filename1 = 'P_Results.xlsx'
input_filename2 = 'RL_Results.xlsx'
output_filename = 'Excel_Analysis.xlsx'


def user_confirm():
    """Function to ask user confirmation"""
    while True:  # Ask user if they want to overwrite the sheet
        bouncer = input("Do you want to overwrite the sheet? (Yes/No): ")
        if bouncer.lower() == "yes":  # If yes, ask for confirmation
            bouncer = input("Are you sure you want to overwrite the Data? (Yes/No): ")
            if bouncer.lower() == "yes":
                return True
            elif bouncer.lower() == "no":
                return False
            else:
                print("Invalid answer. Please respond with 'Yes' or 'No'.")
        elif bouncer.lower() == "no":
            return False
        else:
            print("Invalid answer. Please respond with 'Yes' or 'No'.")


def save_to_excel(df, filename, sheet_name):
    """Function to save DataFrame to Excel file"""
    try:  # Attempt to open an existing workbook
        book = openpyxl.load_workbook(filename)
    except FileNotFoundError:  # If it does not exist, create a new workbook
        book = openpyxl.Workbook()
        book.save(filename)
    if sheet_name in book.sheetnames:  # check if sheet_name already exists
        if book[sheet_name].max_row > 1:  # check if sheet contains data (not empty)
            if not user_confirm():  # Ask user for permission to overwrite
                sheet_name += '_new'  # Modify sheet_name if user does not want to overwrite
    # Write DataFrame to the Excel file
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        if sheet_name in writer.sheets:
            startrow = writer.sheets[sheet_name].max_row
            df.to_excel(writer, index=False, startrow=startrow, sheet_name=sheet_name)


def extract_data(input_filename):
    """Extract Required Data from Excel Files"""
    data_list = []  # List to Store data in
    xls = pd.ExcelFile(input_filename)  # Read Data from Excel File
    for sheet_name in xls.sheet_names:  # Go Through every Tab (Sheet) in Excel File
        df = xls.parse(sheet_name)  # Read the current Tab
        value1 = sheet_name  # Name from Sheet cause Sheet = Instance
        value2 = df.iloc[20, 10]  # Value from K22
        data_list.append([value1, value2])  # add data to list
    return data_list


# Extract data for both excel files
data1 = extract_data(input_filename1)
data2 = extract_data(input_filename2)
# Convert data to pandas DataFrame
df_output1 = pd.DataFrame(data1, columns=['Instance', 'SP Result'])
df_output2 = pd.DataFrame(data2, columns=['Instance', 'RL Result'])
# Merge Instances to get 1 Column only IF the names are identical to prevent mistakes
df_output = pd.merge(df_output1, df_output2, on='Instance')
# Save the merged DataFrame to the output file
save_to_excel(df=df_output, filename=output_filename, sheet_name='Results')
