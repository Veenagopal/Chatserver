from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests

app = FastAPI()

TWO_FACTOR_API_KEY = "c73eba98-66e1-11f0-a562-0200cd936042"

@app.get("/")
def root():
    return "Chat Server is running"

# Request body for sending OTP
class SendOTPRequest(BaseModel):
    phone: str

@app.post("/send-otp")
def send_otp(request_data: SendOTPRequest):
    phone = request_data.phone
    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/{phone}/AUTOGEN"
    response = requests.get(url)

    if response.status_code == 200:
        session_id = response.json().get("Details")
        return {"session_id": session_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to send OTP")

# Request body for verifying OTP
class VerifyOTPRequest(BaseModel):
    session_id: str
    otp: str

@app.post("/verify-otp")
def verify_otp(request_data: VerifyOTPRequest):
    session_id = request_data.session_id
    otp = request_data.otp
    url = f"https://2factor.in/API/V1/{TWO_FACTOR_API_KEY}/SMS/VERIFY/{session_id}/{otp}"
    response = requests.get(url)

    if response.status_code == 200:
        status = response.json().get("Details")
        if status == "OTP Matched":
            return {"message": "OTP verified"}
        else:
            raise HTTPException(status_code=400, detail="OTP incorrect")
    else:
        raise HTTPException(status_code=500, detail="Verification failed")
