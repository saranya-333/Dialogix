import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

SHEET_NAME = "hr_chat_answers"
WORKSHEET_NAME = "Sheet1"

def get_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "service.json", scope
    )
    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)

def log_answer(question: str, answer: str):
    try:
        sheet = get_sheet()
        sheet.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            answer
        ])
        print(" Logged to Google Sheets")

    except Exception as e:
        print("Sheets logging failed (non-fatal)")
        print("Reason:", e)