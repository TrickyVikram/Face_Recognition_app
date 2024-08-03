import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


# Specify the path to your Firebase service account JSON file
json_file_path = "./ServiceAccountKey.json"
databaseURL='https://collegeattendence-fff29-default-rtdb.firebaseio.com/'

# Firebase Initialization
try:
    with open(json_file_path) as f:
        cred = credentials.Certificate(json_file_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': databaseURL,
            'storageBucket': 'collegeattendence-fff29.appspot.com'
        })

except FileNotFoundError:
 print(f"File not found: {json_file_path}. Please check the file path and try again.")
 exit()


    # Reference to the database path where you want to insert data
ReferencePath = db.reference('Students')

    # Data to be inserted
data = {
        "11212652": {
            "name": "Abhijeet kumar",
            "roll": 11212652,
            "Batch": "2021_2025",
            "total_attendence": 0,
            "cource": "B.tech",
            "branch": "CSE",
            "section": "d",
            "last_attendence": "2021-09-01 11:00:15",
        },
         "11212662": {
            "name": "krishna Kumar",
            "roll": 11212662,
            "Batch": "2021_2025",
            "total_attendence": 19,
            "cource": "B.tech",
            "branch": "CSE",
            "section": "c",
            "last_attendence": "2021-09-01 11:00:15",
        },
         "11212820": {
            "name": "Beauty Kumari",
            "roll": 11212820,
            "Batch": "2021_2025",
            "total_attendence": 26,
            "cource": "B.tech",
            "branch": "CSE",
            "section": "f",
            "last_attendence": "2021-09-01 11:00:15",
        }
}

    # Insert data into Firebase Realtime Database
for key, value in data.items():
        ReferencePath.child(key).set(value)
        print(f"\n{key} Data Inserted Successfully in Firebase Realtime Database")
