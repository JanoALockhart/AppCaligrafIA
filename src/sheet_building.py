from io import BytesIO
import xlsxwriter

ROW_HEIGHT_PX = 32
SECOND_COLUMN_WIDTH = 708
FIRST_COLUMN_WIDTH = 36

def build_template_sheet():
    output = BytesIO()

    build_letter_rows_sheet(output, name="template", letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    output.seek(0)

    return output

def build_recommendation_sheet(letters, amount):
    output = BytesIO()

    build_letter_rows_sheet(output, name="template", letters=letters)
    output.seek(0)

    return output

def build_letter_rows_sheet(file_obj, name, letters):
    workbook = xlsxwriter.Workbook(file_obj, {"in_memmory":True})
    worksheet = workbook.add_worksheet(name)

    letter_format = workbook.add_format({
        "font_name": "Arial",
        "font_size": 19,
        "border": 0,
        "top": 1,
        "bottom": 1,
        "align": "left"
    })

    worksheet.set_column_pixels(0, 0, FIRST_COLUMN_WIDTH)
    worksheet.set_column_pixels(1, 1, SECOND_COLUMN_WIDTH)

    row = 0
    for letter in letters:
        
        worksheet.set_row_pixels(row, ROW_HEIGHT_PX)

        worksheet.write(row, 0, letter, letter_format)
        worksheet.write(row, 1, "", letter_format)

        row += 1


    workbook.close()